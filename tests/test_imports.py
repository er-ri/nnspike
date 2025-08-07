#!/usr/bin/env python3
"""Test file to verify nnspike imports"""

# Test imports from nnspike
try:
    from nnspike.constants import Mode
    print(f"Mode.FORWARD = {Mode.FORWARD}")
    print(f"Mode.SMALL_TURN_LEFT = {Mode.SMALL_TURN_LEFT}")
    print("✅ nnspike.constants imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")

try:
    from nnspike.models import NvidiaModel
    print("✅ nnspike.models imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")

try:
    from nnspike.unit import ETRobot
    print("✅ nnspike.unit imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")

print("Test completed!")
