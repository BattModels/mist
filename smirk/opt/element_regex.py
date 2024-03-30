import fileinput

leader = None
options = []
element_regex = []
has_leader_only = False  # If true match `X[...]?` otherwise match `X[...]`
for line in fileinput.input():
    line = line.strip()
    assert len(line) <= 2

    # Same leader, new option
    if leader == line[0] and len(line) == 2:
        options.append(line[1])

    # Flush current Regex
    elif leader is not None:
        leader_regex = leader
        if options:
            options = list(set(options))
            options.sort()
            leader_regex += f"[{'|'.join(options)}]"

            if not has_leader_only:
                leader_regex += "?"

        element_regex.append(leader_regex)
        leader = None
        options = []

    # New leader
    if leader is None:
        leader = line[0]
        has_leader_only = False
        if len(line) == 2:
            has_leader_only = True
            options.append(line[1])


# Generate Code
print("Split on Leader Character:")
for idx, r in enumerate(element_regex):
    sep = "|" if idx != len(element_regex) - 1 else ""
    print(f'r"{r}{sep}",')

print("\nFull Regex:")
print("|".join(element_regex))
