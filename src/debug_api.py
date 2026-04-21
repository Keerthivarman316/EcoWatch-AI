import api
import asyncio
import traceback

async def debug():
    try:
        print("Starting manual startup...")
        await api.startup_event()
        print("Startup successful!")
    except Exception:
        print("Error during startup:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug())
