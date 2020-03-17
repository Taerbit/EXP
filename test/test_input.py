import EXP

class basic_container:
    """Class to mimic the behaivour of a container class"""

    def __init__(self, numbers, children=[], ordered=True):
        self.numbers = numbers
        self.counter = 0
        self.children = children
        self.ordered = ordered

    def data_remaining(self):
        if self.counter == len(self.numbers):
            return False
        return True

    def load(self):
        n = self.numbers[self.counter]
        self.increment()
        return n

    def get_number(self):
        if self.ordered:
            return self.numbers[self.counter]
        else:
            return -1

    def increment(self):
        self.counter = self.counter + 1
        if not self.children == []:
            for c in self.children:
                if not c.increment():
                    return False
        return self.data_remaining()


def test_sorter_basic():
    container = [
        basic_container([1, 2, 3, 4, 5, 6]),
        basic_container([1, 2, 4, 5])
    ]
    s = EXP.src.Input.Sorter(container)

    assert s.get_id() == 1
    assert s.has_next()
    assert s.get_next() == [1, 1]
    assert s.get_next() == [2, 2]
    assert s.get_next() == [4, 4]
    assert s.get_next() == [5, 5]
    assert not s.has_next()

def test_sorter_unordered():
    container = [
        basic_container([1, 2, 3, 4, 5, 6]),
        basic_container([1, 2, 4, 5]),
        basic_container([1, 2, 3, 4, 5], ordered=False)
    ]
    s = EXP.src.Input.Sorter(container)

    assert s.get_id() == 1
    assert s.has_next()
    assert s.get_next() == [1, 1, 1]
    assert s.get_next() == [2, 2, 2]
    assert s.get_next() == [4, 4, 3]
    assert s.get_next() == [5, 5, 4]
    assert not s.has_next()