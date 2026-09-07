"""Publish explicit contact-channel state without calling silence "no collision"."""

import json
import time
import uuid

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from ros_gz_interfaces.msg import Contacts
from std_msgs.msg import String


class ContactEvidenceNode(Node):
    def __init__(self):
        super().__init__('contact_evidence_node')
        self.declare_parameter('contact_topic', '/world_contacts')
        self.declare_parameter('status_topic', '/sim/contact_channel_status')
        self.declare_parameter('source_ids', [''])
        self.declare_parameter('heartbeat_s', 1.0)
        contact_topic = str(self.get_parameter('contact_topic').value)
        status_topic = str(self.get_parameter('status_topic').value)
        self.source_ids = tuple(
            value for value in (str(v).strip() for v in self.get_parameter('source_ids').value)
            if value
        )
        heartbeat_s = float(self.get_parameter('heartbeat_s').value)
        if not heartbeat_s > 0.0:
            raise RuntimeError('contact heartbeat_s must be positive')
        self.epoch = uuid.uuid4().hex
        self.sequence = 0
        self.delivery_count = 0
        self.contact_count = 0
        self.last_source_stamp_ns = None
        self.last_receipt_wall_s = None
        self._publisher = self.create_publisher(String, status_topic, 10)
        self.create_subscription(Contacts, contact_topic, self._contact_cb, 10)
        self.create_timer(
            heartbeat_s, self._heartbeat,
            clock=Clock(clock_type=ClockType.SYSTEM_TIME),
        )
        self._heartbeat()

    def _contact_cb(self, message: Contacts) -> None:
        self.delivery_count += 1
        self.contact_count += len(message.contacts)
        self.last_receipt_wall_s = time.time()
        stamp = message.header.stamp
        self.last_source_stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _heartbeat(self) -> None:
        self.sequence += 1
        if not self.source_ids:
            state = 'missing_configuration'
        elif self.delivery_count:
            state = 'contact_deliveries_observed'
        else:
            # Contact sensors are event-only. This state deliberately does not
            # claim either channel health or a collision-free physical state.
            state = 'configured_silent_requires_positive_control'
        payload = {
            'schema_version': 1,
            'producer_epoch': self.epoch,
            'event_id': f'{self.epoch}:{self.sequence}',
            'sequence': self.sequence,
            'state': state,
            'source_ids': self.source_ids,
            'configured_source_count': len(self.source_ids),
            'delivery_count': self.delivery_count,
            'contact_count': self.contact_count,
            'last_source_stamp_ns': self.last_source_stamp_ns,
            'last_receipt_wall_s': self.last_receipt_wall_s,
            'silence_is_no_contact': False,
        }
        message = String()
        message.data = json.dumps(payload, separators=(',', ':'))
        self._publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = ContactEvidenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        finally:
            try:
                if rclpy.ok():
                    rclpy.shutdown()
            except RuntimeError:
                pass


if __name__ == '__main__':
    main()
