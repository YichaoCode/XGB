import csv
import statistics


def read_data(filename):
    with open(filename, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        data = {field: [] for field in reader.fieldnames}
        for row in reader:
            for field in reader.fieldnames:
                value = row[field]
                if field == 'materials':
                    data[field].append(value)
                else:
                    data[field].append(float(value))
    return data


def compute_stats(data):
    stats = {}
    for field, values in data.items():
        if field == 'materials':
            continue
        mean = statistics.mean(values)
        median = statistics.median(values)
        stdev = statistics.stdev(values)
        min_val = min(values)
        max_val = max(values)
        stats[field] = {
            'mean': mean,
            'median': median,
            'stdev': stdev,
            'min': min_val,
            'max': max_val,
        }
    return stats


def main():
    data = read_data('Train.csv')
    stats = compute_stats(data)
    print(f"{'feature':<10} {'mean':>10} {'median':>10} {'stdev':>10} {'min':>10} {'max':>10}")
    for field in stats:
        s = stats[field]
        print(f"{field:<10} {s['mean']:10.4f} {s['median']:10.4f} {s['stdev']:10.4f} {s['min']:10.4f} {s['max']:10.4f}")


if __name__ == '__main__':
    main()
