import subprocess
import json
import os
import re # Added for parsing web URLs if needed

def get_modified_files_from_commit(commit_hash: str, repo_path: str = '.', remote_url: str = None, remote_name: str = 'external_source') -> list | None:
    original_cwd = os.getcwd()
    try:
        os.chdir(repo_path)
        #print(f"Working in Git repository at: {os.getcwd()}")

        # 1. Add and fetch from remote if URL is provided
        if remote_url:
            #print(f"Attempting to manage remote '{remote_name}' with URL: {remote_url}...")
            try:
                # Check if remote already exists and has the correct URL
                result_check_remote = subprocess.run(
                    ['git', 'remote', 'get-url', remote_name],
                    capture_output=True, text=True, check=False
                )
                existing_url = result_check_remote.stdout.strip()

                if result_check_remote.returncode == 0 and existing_url == remote_url:
                    #print(f"Remote '{remote_name}' already exists with the correct URL.")
                    pass
                elif result_check_remote.returncode == 0 and existing_url != remote_url:
                    #print(f"Remote '{remote_name}' exists with a different URL ({existing_url}). Updating to {remote_url}.")
                    subprocess.run(
                        ['git', 'remote', 'set-url', remote_name, remote_url],
                        check=True, capture_output=True, text=True
                    )
                    #print(f"Remote '{remote_name}' URL updated.")
                else: # Remote not found, add it
                    subprocess.run(
                        ['git', 'remote', 'add', remote_name, remote_url],
                        check=True,
                        capture_output=True, text=True
                    )
                    print(f"Remote '{remote_name}' added successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error managing remote '{remote_name}': {e.stderr.strip()}")
                return None # Critical error setting up remote

            #print(f"Fetching all branches and tags from remote '{remote_name}' to ensure commit objects are present...")
            try:
                # **KEY CHANGE HERE:** Fetch all branches and tags for robustness
                subprocess.run(
                    ['git', 'fetch', remote_name, '--tags', '--force'],
                    check=True,
                    capture_output=True, text=True
                )
                #print(f"Fetched from '{remote_name}' successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error fetching from remote '{remote_name}': {e.stderr.strip()}")
                return None

        # 2. Get the modified files using git show --name-only
        #print(f"Getting file list for commit: {commit_hash}")
        try:
            # Use --name-only to get just the file paths
            # --pretty=format: suppresses commit message and other header details
            command = ['git', 'show', '--name-only', '--pretty=format:', commit_hash]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True  # Will raise CalledProcessError if git command fails (e.g., bad object)
            )

            # Split the output into lines and filter out empty ones
            file_list = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            return file_list

        except subprocess.CalledProcessError as e:
            # Check for "bad object" specifically
            if "fatal: bad object" in e.stderr:
                print(f"Error: Commit '{commit_hash}' not found or is a bad object in the local repository.")
                print("This might happen if the commit was force-pushed over or deleted from the remote.")
            else:
                print(f"Error executing git show for commit {commit_hash}:")
                print(f"  Return Code: {e.returncode}")
                print(f"  STDOUT: {e.stdout.strip()}")
                print(f"  STDERR: {e.stderr.strip()}")
            return None
        except FileNotFoundError:
            print("Error: 'git' command not found. Make sure Git is installed and in your PATH.")
            return None
    finally:
        os.chdir(original_cwd) # Always return to the original working directory

