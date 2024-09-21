from collections import Counter

class Distribution(Counter):
    def __mul__(self, other):
        if isinstance(other, Counter):
            # NB: if changing the distribution as below to allow for addition
            # between disjoint sets then perhaps we want to change it here as well.
            if self.keys() != other.keys():
                raise ValueError("Distributions must have the same keys.")
            return Distribution({k: v * other[k] for k, v in self.items()})
        elif isinstance(other, int):  
            # NB: Change the implementation of multiplication with integers
            return Distribution({k: v * other for k, v in self.items()})
        else:
            raise ValueError("Unsupported operand for * : '{}'' and '{}'".format(
                self.__class__, type(other)))

    def __add__(self, other):
        '''Add counts from two distributions.

        '''
        if isinstance(other, Counter):
            result = Distribution()
            for elem, count in self.items():
                newcount = count + (other[elem] if elem in other else 0)
                result[elem] = newcount
            for elem, count in other.items():
                if elem not in self:
                    result[elem] = count
            return result
        elif isinstance(other, int):  
            return Distribution({k: v + other for k, v in self.items()})
        else:
            return NotImplemented

    def __sub__(self, other):
        '''Subtracts counts from two distributions.

        '''
        if isinstance(other, Counter):
            result = Distribution()
            for elem, count in self.items():
                newcount = count - (other[elem] if elem in other else 0)
                result[elem] = newcount
            for elem, count in other.items():
                if elem not in self:
                    result[elem] = -count
            return result
        elif isinstance(other, int):  
            return Distribution({k: v - other for k, v in self.items()})
        else:
            return NotImplemented

    def normalize(self, uniform_prior=False):
        default = 0
        if uniform_prior:
            default = 1 / len(self)
        total = sum(self.values())
        return Distribution({k : (v / total) if total != 0 else default
                for k, v in self.items()})

    def __repr__(self):
        if not self:
            return f'{self.__class__.__name__}()'
        try:
            # Sort by the keys (opposite of what Counter does)
            d = {k: self[k] for k in sorted(self)}
        except TypeError:
            # handle case where values are not orderable
            d = dict(self)
        return f'{self.__class__.__name__}({d!r})'

    def apply(self, function):
        result = Distribution()
        for elem, count in self.items():
            result[elem] = function(count)
        return result
