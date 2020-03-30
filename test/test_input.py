import EXP

class basic_container:
    """Class to mimic the behaivour of a container class"""

    def __init__(self, numbers, name, children=[], ordered=True):
        self.fp = numbers
        self.counter = 0
        self.children = children
        self.ordered = ordered
        self.name = name

    def data_remaining(self):
        if self.counter == len(self.fp):
            return False
        return True

    def load(self):
        n = self.fp[self.counter]
        self.increment()
        return n

    def get_number(self):
        if self.ordered:
            return self.fp[self.counter]
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
        basic_container([1, 2, 3, 4, 5, 6], "basic1"),
        basic_container([1, 2, 4, 5], "basic2"),
        basic_container([1, 2, 3, 4, 5], "basic3")
    ]
    s = EXP.src.Input.Sorter(container, 0)

    assert s.has_next()
    assert s.get_next() == ({"basic1": 1, "basic2": 1, "basic3": 1}, '1')
    assert s.get_next() == ({"basic1": 2, "basic2": 2, "basic3": 2}, '2')
    assert s.get_next() == ({"basic1": 4, "basic2": 4, "basic3": 4}, '4')
    assert s.get_next() == ({"basic1": 5, "basic2": 5, "basic3": 5}, '5')
    assert not s.has_next()

def test_sorter_unordered():
    container = [
        basic_container([1, 2, 3, 4, 5, 6], "basic1"),
        basic_container([1, 2, 4, 5], "basic2"),
        basic_container([1, 2, 3, 4, 5], "basic3", ordered=False)
    ]
    s = EXP.src.Input.Sorter(container)

    assert s.has_next()
    assert s.get_next() == ({"basic1": 1, "basic2": 1, "basic3": 1}, '1')
    assert s.get_next() == ({"basic1": 2, "basic2": 2, "basic3": 2}, '2')
    assert s.get_next() == ({"basic1": 4, "basic2": 4, "basic3": 3}, '4')
    assert s.get_next() == ({"basic1": 5, "basic2": 5, "basic3": 4}, '5')
    assert not s.has_next()

def test_sorter_advanced():
    container = [
        basic_container([1, 4, 5, 6, 7, 10], "basic1"),
        basic_container([1, 2, 4, 5, 10], "basic2"),
        basic_container([1, 3, 4, 5, 10], "basic3")
    ]
    s = EXP.src.Input.Sorter(container, id_index=1)

    assert s.has_next()
    s.get_next()
    assert s.get_next() == ({"basic1": 4, "basic2": 4, "basic3": 4}, '4')
    assert s.get_next() == ({"basic1": 5, "basic2": 5, "basic3": 5}, '5')
    assert s.get_next() == ({"basic1": 10, "basic2": 10, "basic3": 10}, '10')
    assert not s.has_next()