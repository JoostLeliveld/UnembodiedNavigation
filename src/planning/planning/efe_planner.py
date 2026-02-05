#!/usr/bin/env python3
"""Legacy shim for efe_planner.

The monolithic planner has been split into clean planner classes and thin ROS nodes.
This module remains for backwards compatibility and forwards to the EFE2 node.
"""

from planning.nodes.efe2_planner_node import main


if __name__ == '__main__':
    main()
