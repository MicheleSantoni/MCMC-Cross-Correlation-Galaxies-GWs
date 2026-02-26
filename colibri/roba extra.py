def load_params_from_file(self, filename):
    print(f"Reading from file: {filename}")
    with open(filename, 'r') as f:
        for line in f:
            # print("Raw line:", repr(line))  # DEBUG: mostra spazi invisibili
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                parsed_value = self._parse_value(value)
                # print(f"Storing param: {key} = {parsed_value}")
                self.params[key] = parsed_value


def _parse_value(self, val):
    val = val.strip()
    if val.lower() in ['yes', 'true']:
        return 'yes'
    if val.lower() in ['no', 'false']:
        return 'no'
    try:
        if ',' in val:
            return [float(x) for x in val.split(',')]
        return int(val) if '.' not in val else float(val)
    except ValueError:
        return val