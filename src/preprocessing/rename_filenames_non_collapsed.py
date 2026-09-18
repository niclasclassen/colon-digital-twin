import re
from pathlib import Path

pattern = re.compile(r"^.*?(\d+)_pos-(prone|supine).*\.mha$")


for line in Path("data/raw/filenames_non_collapsed_original.txt").read_text().splitlines():
    m = pattern.match(line)

    print(m)
    if not m:
        print(f"Skipping: {line}")
        continue

    number = int(m.group(1))
    position = m.group(2)
    new_name = f"colon_{number:04d}-{position}.mha"
    print(f"{line} -> {new_name}")
    with open("data/raw/filenames_non_collapsed_renamed.txt", "a") as file:
        file.write(new_name + "\n")

