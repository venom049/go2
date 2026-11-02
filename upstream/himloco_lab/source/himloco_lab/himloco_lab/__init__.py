# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HIMLoco components; importing the package does not load simulator tasks.

The Go2 stair project defines its own task and assets.  Applications wishing to
use the original demonstration tasks can call :func:`register_tasks` after
launching Isaac Sim and configuring those tasks' asset paths.
"""


def register_tasks():
    """Explicitly register the upstream demonstration tasks (requires Isaac Lab)."""
    from . import tasks

    return tasks
