class MetricF1:

    def __init__(self):
        self.clear()

    def clear(self):
        self.p = 0
        self.total_pr = 0
        self.total_re = 0

    def update(self, predictions):
        for item in predictions:
            if item["pred"] != []:
                self.total_pr += 1
            if item["target"] != []:
                self.total_re += 1
            if item["pred"] != [] and item["pred"] == item["target"]:
                self.p += 1

        return self.f1(), self.pr(), self.re()

    def pr(self):
        if self.total_re == 0:
            return 1
        return self.p / self.total_pr if self.total_pr != 0 else 0

    def re(self):
        return self.p / self.total_re if self.total_re != 0 else 1

    def f1(self):
        return 2 / (1 / self.pr() + 1 / self.re()) if self.pr() + self.re() != 0 else 0
