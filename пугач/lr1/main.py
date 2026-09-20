
b = [1, 2, 3, 4, 5, 6, "213", "2eds", {"a": "str", "t": "a"}]

for el in b:
    if isinstance(el, dict):
        for key, value in el.items():
            print(f"{key}: {value}")
    else:
        print(el)
