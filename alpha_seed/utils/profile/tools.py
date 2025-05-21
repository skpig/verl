from typing import List
try:
    import ujson as json
except ImportError:
    import json
import gzip
import os


def merge_trace(files: List[str]):
    # Initialize an empty list to store all events
    all_events = []

    # Iterate over all files in the input list
    for file_path in files:
        print('start to process file: ', file_path, '...')
        filename = os.path.basename(file_path)
        if filename.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
        elif filename.endswith('.gz'):
            with gzip.open(file_path, 'rt') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file type: {filename}")

        if isinstance(data, list):
            all_events.extend(data)
        elif isinstance(data, dict) and 'traceEvents' in data:
            display_time_unit = data.get('displayTimeUnit', 'ms')
            base_time_us = data.get('baseTimeNanoseconds', 0) / 1e6
            ts_to_us_coefficient = 1  # 将event中的ts转换为us需要乘以的系数
            if display_time_unit == 'ms':
                ts_to_us_coefficient = 1e3
            elif display_time_unit == 'us':
                ts_to_us_coefficient = 1
            else:
                # 遇到其他的单位时再加
                raise NotImplementedError

            # convert event ts to us
            for e in data['traceEvents']:
                e['ts'] += e['ts'] * ts_to_us_coefficient + base_time_us
            all_events.extend(data['traceEvents'])

    # Convert the list of events to a JSON string
    json_data = json.dumps(all_events)

    # Compress the JSON string and save it as a gzipped file
    output_file_path = os.path.join('combined_events.json.gz')
    with gzip.open(output_file_path, 'wt') as f:
        f.write(json_data)

    print(f"All events have been combined and saved to {output_file_path}")


if __name__ == '__main__':
    files = [
        '/tmp/ray-timeline-2025-04-15_15-56-44.json',
        'trace.json.gz',
    ]
    merge_trace(files)
