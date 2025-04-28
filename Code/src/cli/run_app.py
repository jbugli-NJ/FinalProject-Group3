"""
Wraps a CLI abstraction around the 'streamlit run' command.
"""

# %% Imports

import os
import subprocess
import sys

# %%

def main():
    """
    Runs the streamlit app, abstracting the 'streamlit run' command.
    """

    # Construct the path to to the app execution script
    # TODO: Argparse

    app_path = os.path.join('src', 'app', 'app.py')

    app_path = os.path.abspath(app_path)

    if not os.path.exists(app_path):
         print(f"Cannot find app.py at: {app_path}")
         sys.exit(1)


    # Construct and execute a subprocess command

    command = ['streamlit', 'run', app_path]
    print(f"Executing command: {' '.join(command)}")

    try:
        subprocess.run(command, check=True)

    except FileNotFoundError:
         print("'streamlit' command not found!")
         sys.exit(1)

    except subprocess.CalledProcessError as e:
         print(f"Error running Streamlit: {e}")
         sys.exit(e.returncode)

    except KeyboardInterrupt:
         print("\nStreamlit server stopped.")
         sys.exit(0)


if __name__ == '__main__':
    main()