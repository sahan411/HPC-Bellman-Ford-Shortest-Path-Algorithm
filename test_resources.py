#!/usr/bin/env python3
import sys
sys.path.insert(0, 'd:\\HPC')

from app import get_system_resources
result = get_system_resources()
print("Direct call result:")
print(result.get_json())
