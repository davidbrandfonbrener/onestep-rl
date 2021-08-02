
import csv
from collections import defaultdict


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MeterGroup(object):
    def __init__(self, csv_path=None):
        self._meters = defaultdict(AverageMeter)
        self._meter_groups = defaultdict(MeterGroup)
        self._csv_file = None
        if csv_path is not None:
            self._csv_file = open(csv_path, 'w', newline='')
            self._csv_writer = None

    def update(self, meter_path, value):
        split_path = meter_path.split('/')
        head = split_path[0]
        if len(split_path) == 1:
            self._meters[head].update(value)
        else:
            tail = '/'.join(split_path[1:])
            self._meter_groups[head].update(tail, value)

    def update_dict(self, input_dict):
        for k,v in input_dict.items():
            self.update(k, v)

    def value(self, meter_path):
        split_path = meter_path.split('/')
        head = split_path[0]
        if len(split_path) == 1:
            return self._meters[head].value()
        else:
            tail = '/'.join(split_path[1:])
            return self._meter_groups[head].value(tail)

    def values(self):
        values = {}
        for meter_name, meter in self._meters.items():
            values[meter_name] = meter.value()
        for mg_name, mg in self._meter_groups.items():
            for k, v in mg.values().items():
                values[mg_name + '/' + k] = v
        return values

    def write_csv(self):
        if self._csv_file is None:
            for mg in self._meter_groups.values():
                mg.write_csv()
            return

        values = self.values()
        if self._csv_writer is None:
            fields = sorted(values.keys())
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=fields, restval='NA')
            self._csv_writer.writeheader()
        self._csv_writer.writerow(values)
        self._csv_file.flush()

    def write_console(self, format_str):
        values = self.values()
        print(format_str.format(values=values))

    def clear(self):
        self._meters.clear()
        for mg in self._meter_groups.values():
            mg.clear()


class Logger(MeterGroup):
    # needs to create sub-meter-groups that have csv files
    # and keep track of format strings
    def setup(self, sub_meter_map, summary_format_str=None):
        self.summary_format_str = summary_format_str
        self.sub_meter_map = sub_meter_map
        for k, mg_spec in self.sub_meter_map.items():
            mg = MeterGroup(mg_spec['csv_path'])
            self._meter_groups[k] = mg

    def write_console(self):
        if self.summary_format_str is None:
            for mg_name, mg_spec in self.sub_meter_map.items():
                self._meter_groups[mg_name].write_console(
                    mg_spec['format_str'])
        else:
            super().write_console(self.summary_format_str)

    def write_all(self):
        self.write_console()
        self.write_csv()
        self.clear()

    def write_sub_meter(self, sub_meter):
        self._meter_groups[sub_meter].write_csv()
        self._meter_groups[sub_meter].write_console(
            self.sub_meter_map[sub_meter]['format_str'])
        self._meter_groups[sub_meter].clear()


default_logger = Logger()