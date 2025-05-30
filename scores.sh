Here is the updated shell script that will execute the Python commands for each name, and then commit all changes in the current directory to a GitHub repository with the message "results" and push them.

#!/bin/bash

# List of names to iterate through
NAMES=(
    #"deepseek-r1-0528" reasoning ahh
    #"qwen3-32b-fp8" reasoning ahh
    #"lfm-40b" no log probs, will think about how to patch
    #"hermes3-8b" already did
    # "llama3.2-3b-instruct"
    # "llama3.1-405b-instruct-fp8"
    "llama3.3-70b-instruct-fp8"
    "hermes3-405b"
    # "llama3.2-11b-vision-instruct"
    #"qwen25-coder-32b-instruct" i bet its going to try to reason
    # "llama-4-maverick-17b-128e-instruct-fp8"
    "deepseek-llama3.3-70b"
    #"deepseek-r1-671b" reasoning model
    "llama3.1-8b-instruct"
    "deepseek-v3-0324"
    # "llama3.1-nemotron-70b-instruct-fp8"
    #"lfm-7b" no log probs?
)

# Iterate over each name in the array
for NAME in "${NAMES[@]}"; do
    echo "--- Running for NAME: $NAME ---"
    
    # Execute the first Python script
    echo "Executing: python3 generate_summaries.py $NAME 350"
    python3 generate_summaries.py "$NAME" 350
    
    # Check if the first script executed successfully
    if [ $? -ne 0 ]; then
        echo "Error: generate_summaries.py failed for $NAME. Skipping experiments.py."
        continue # Move to the next NAME if the first script fails
    fi
        
    # Add all changes to the staging area
    git add .

    # Commit the changes with the specified message
    git commit -m "results"

    # Push the changes to the remote repository
    git push

    echo "--- Git operations completed ---"

    # Execute the second Python script
    echo "Executing: python3 experiments.py $NAME 350 compare"
    python3 experiments.py "$NAME" 350 compare
    
    # Check if the second script executed successfully
    if [ $? -ne 0 ]; then
        echo "Error: experiments.py failed for $NAME."
        # Decide if you want to stop the entire script or continue
        # For now, we'll just log the error and continue to the next name.
    fi
        
    # Add all changes to the staging area
    git add .

    # Commit the changes with the specified message
    git commit -m "results for $NAME"

    # Push the changes to the remote repository
    git push

    echo "--- Git operations completed ---"

    echo "" # Add a blank line for readability between runs
done

echo "--- All Python scripts finished. Now committing to Git ---"
