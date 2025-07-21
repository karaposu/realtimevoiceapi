import sys
import os
sys.path.append(os.path.dirname(__file__))

import asyncio
from realtimevoiceapi.smoke_tests.test_10_metrics_and_cleanup import test_resource_limits

if __name__ == "__main__":
    asyncio.run(test_resource_limits())