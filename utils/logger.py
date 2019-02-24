from sys import stderr, exc_info


class Logger:
    def __init__(self, folder):
        self._lines = []

        self._filepath = folder + "/log.txt"

        try:
            self._file = open(self._filepath, "w")
        except Exception:
            print("Can't open log file!", file=stderr)
            ex_type, ex_inst, ex_tb = exc_info()
            raise ex_type.with_traceback(ex_inst, ex_tb)

    def log(self, line):
        self._lines.append(line)

        if len(self._lines) > 100:
            self._file.write("\n".join(self._lines) + "\n")
            self._lines.clear()

    def finish(self):
        self._file.write("\n".join(self._lines))
        self._file.close()
        self._file = None
