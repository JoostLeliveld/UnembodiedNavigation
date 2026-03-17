import math

from std_msgs.msg import Float64MultiArray


DETECTION_DIAGNOSTICS_TOPIC = '/perception/detection_diagnostics'

DETECTION_DIAGNOSTIC_FIELDS = (
    'stamp',
    'detected',
    'u_mid',
    'v_mid',
    'yaw_est',
    'u_red',
    'v_red',
    'red_area_px',
    'u_blue',
    'v_blue',
    'blue_area_px',
    'separation_px',
    'border_margin_px',
)

_FIELD_INDEX = {
    name: idx for idx, name in enumerate(DETECTION_DIAGNOSTIC_FIELDS)
}


def diagnostics_message(
    *,
    stamp,
    detected,
    u_mid,
    v_mid,
    yaw_est,
    u_red,
    v_red,
    red_area_px,
    u_blue,
    v_blue,
    blue_area_px,
    separation_px,
    border_margin_px,
):
    msg = Float64MultiArray()
    msg.data = [
        float(stamp),
        1.0 if detected else 0.0,
        float(u_mid),
        float(v_mid),
        float(yaw_est),
        float(u_red),
        float(v_red),
        float(red_area_px),
        float(u_blue),
        float(v_blue),
        float(blue_area_px),
        float(separation_px),
        float(border_margin_px),
    ]
    return msg


def diagnostics_from_message(msg: Float64MultiArray):
    data = list(msg.data)
    if len(data) < len(DETECTION_DIAGNOSTIC_FIELDS):
        data.extend([math.nan] * (len(DETECTION_DIAGNOSTIC_FIELDS) - len(data)))

    values = {
        name: float(data[idx]) for name, idx in _FIELD_INDEX.items()
    }
    values['detected'] = bool(values['detected'] >= 0.5)
    return values
