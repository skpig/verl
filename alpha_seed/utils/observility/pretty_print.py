import datetime
import pprint


class TimePrefixedPrettyPrinter(pprint.PrettyPrinter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _format(self, object, stream, indent, allowance, context, level):
        # Add a timestamp prefix to the top-level elements
        if level == 0:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            stream.write(f"[{current_time}] ")
        super()._format(object, stream, indent, allowance, context, level)


printer = TimePrefixedPrettyPrinter()

pprint = printer.pprint
