#!/bin/bash

# Specify the size limit in bytes.
# On Gitlab, files must be smaller than 5MB.
export SIZE_LIMIT=5242880

# Define the directories to exclude from the search, relative to the current directory
# Separate multiple directories with spaces, e.g., EXCLUDE_DIRS=("dir1" "dir2" "dir3")
EXCLUDE_DIRS=(".git/objects")

# Function to convert bytes to human readable format
human_filesize() {
  awk -v sum=$1 ' BEGIN { hum[1024^3]="GB";hum[1024^2]="MB";hum[1024]="KB"; for (x=1024^3; x>=1024; x/=1024){
    if (sum>=x) { printf "%.2f %s\n", sum/x, hum[x]; break }
  }}'
}

# Export the function so that it can be used in a subshell by 'find'
export -f human_filesize

# Initialize oversized_found flag
oversized_found=0

# Initialize the find command with an array to properly handle spaces and special characters
find_cmd=(find . -type f)

# Add the exclusion patterns to the find command
for dir in "${EXCLUDE_DIRS[@]}"; do
  find_cmd+=(! -path "./$dir/*")
done

# Execute the find command and check file sizes
"${find_cmd[@]}" -exec bash -c '
  for file do
    filesize=$(stat -c%s "$file")
    if (( filesize > SIZE_LIMIT )); then
      human_size=$(human_filesize "$filesize")
      echo "Error: File "$file" is larger than the limit ($human_size)"
      exit 1
    fi
  done
' bash {} + || oversized_found=1

# Exit with a non-zero exit code if an oversized file was found
if (( oversized_found == 1 )); then
  exit 1
fi

# If we get here, no oversized files were found
exit 0
