#!/usr/bin/env python3
# This code is part of Qiskit.
#
# (C) Copyright IBM 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility script to verify that all images have alt text"""

from pathlib import Path
import multiprocessing
import sys
import glob

# Dictionary to allowlist lines of code that the checker will not error
# Format: {"file_path": [list_of_line_numbers]}
ALLOWLIST_MISSING_ALT_TEXT = {
    "qiskit_addon_cutting/instructions/move.py": [50]
}


def is_image(line: str) -> bool:
    return line.strip().startswith((".. image:", ".. plot:"))


def is_option(line: str) -> bool:
    return line.strip().startswith(":")

def in_allowlist(filename: str, line_num: int) -> bool:
    return line_num in ALLOWLIST_MISSING_ALT_TEXT.get(filename, [])


def validate_image(file_path: str) -> tuple[str, list[str]]:
    """Validate all the images of a single file"""
    invalid_images: list[str] = []

    lines = Path(file_path).read_text().splitlines()

    line_index = 0
    image_found = False
    image_line = -1
    options: list[str] = []

    while line_index < len(lines):
        line = lines[line_index].strip()

        if image_found and not is_option(line) and not is_valid_image(options):
            invalid_images.append(f"- Error in line {image_line}: {lines[image_line-1].strip()}")
            image_found = False
            options = []
            continue

        if image_found and is_option(line):
            options.append(line)

        if is_image(line) and not in_allowlist(file_path, line_index + 1):
            image_found = True
            image_line = line_index + 1
            options = []

        line_index += 1

    return (file_path, invalid_images)


def is_valid_image(options: list[str]) -> bool:
    alt_exists = any(option.startswith(":alt:") for option in options)
    nofigs_exists = any(option.startswith(":nofigs:") for option in options)

    # Only `.. plot::`` directives without the `:nofigs:` option are required to have alt text.
    # Meanwhile, all `.. image::` directives need alt text and they don't have a `:nofigs:` option.
    return alt_exists or nofigs_exists

def main() -> None:
    files = glob.glob("qiskit_addon_cutting/**/*.py", recursive=True)
    
    with multiprocessing.Pool() as pool:
        results = pool.map(validate_image, files)

    failed_files = [x for x in results if len(x[1])]

    if not len(failed_files):
        print("✅ All images have alt text")
        sys.exit(0)

    print("💔 Some images are missing the alt text", file=sys.stderr)

    for filename, image_errors in failed_files:
        print(f"\nErrors found in {filename}:", file=sys.stderr)

        for image_error in image_errors:
            print(image_error, file=sys.stderr)

    print(
        "\nAlt text is crucial for making documentation accessible to all users. It should serve the same purpose as the images on the page, conveying the same meaning rather than describing visual characteristics. When an image contains words that are important to understanding the content, the alt text should include those words as well.",
        file=sys.stderr,
    )

    sys.exit(1)


if __name__ == "__main__":
    main()