values = {
}

def assert_no_change(value, key):
    if key not in values:
        values[key] = value
    else:
        assert value == values[key]
