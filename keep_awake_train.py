import sys
import ctypes
import os
import time

# Windows API constants to prevent sleep
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002 # Optional: keeps screen on too

def prevent_sleep():
    """Tells Windows to stay awake."""
    print("--- [KEEP-ALIVE] Telling Windows to stay awake during training... ---")
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

def allow_sleep():
    """Tells Windows it can sleep again."""
    print("--- [KEEP-ALIVE] Releasing sleep prevention. ---")
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)

if __name__ == "__main__":
    try:
        # 1. Prevent Sleep
        prevent_sleep()
        
        # 2. Run the actual training
        print("--- [KEEP-ALIVE] Starting Training... ---\n")
        import train
        train.main()
        
    except KeyboardInterrupt:
        print("\n--- [KEEP-ALIVE] Training interrupted by user. ---")
    except Exception as e:
        print(f"\n--- [KEEP-ALIVE] An error occurred: {e} ---")
    finally:
        # 3. Allow sleep again when done
        allow_sleep()
