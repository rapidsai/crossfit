import abc
from typing import Dict
from dataclasses import field, MISSING, Field


from typing_utils import get_origin


def state(
    init,
    combine=None,
    # TODO: maybe add shape and dtype?
    equals=None,
    is_list=False,
    repr=True,
    hash=None,
    compare=True,
    metadata=None,
):
    metadata = metadata or {}
    if combine is not None:
        if combine in (sum, "sum"):
            combine = _sum
        metadata["combine"] = combine
    if equals is not None:
        metadata["equals"] = equals
    if is_list:
        metadata["is_list"] = is_list

    _init = True
    if callable(init):
        default = MISSING
        default_factory = init
        _init = False
    else:
        default_factory = MISSING
        default = init

    return field(
        default=default,
        default_factory=default_factory,
        init=_init,
        repr=repr,
        compare=compare,
        hash=hash,
        metadata=metadata,
    )


class CrossModule:
    def setup(self, **kwargs):
        for name, state_field in self.fields().items():
            if name in kwargs:
                setattr(self, name, kwargs[name])
            else:
                if field.default_factory is not MISSING:
                    setattr(self, name, state_field.default_factory())
                else:
                    setattr(self, name, state_field.default)
        self._setup = True

    @classmethod
    def fields(cls) -> Dict[str, Field]:
        output = {}
        for name in dir(cls):
            if not name.startswith("_"):
                part = getattr(cls, name)
                if isinstance(part, Field):
                    output[name] = part
                    part.name = name

        return output

    def combine(self, other):
        merged_fields = {}

        for state_field in self.fields().values():
            _type = get_origin(state_field.type)
            _type = _type or state_field.type

            if "combine" in state_field.metadata:
                merged_fields[state_field.name] = state_field.metadata["combine"](
                    getattr(self, field.name),
                    getattr(other, field.name),
                )
            # TODO: Fix this
            elif issubclass(_type, CrossModule):
                # TODO: Add validation
                child = getattr(self, state_field.name)
                merged_fields[state_field.name] = child.combine(
                    getattr(other, state_field.name),
                )
            else:
                raise ValueError(
                    f"Cannot combine {self.__class__.__name__} automatically "
                    + "please implement a combiner function: `def combine(self, other)`"
                )

        return self.__class__(**merged_fields)

    def __add__(self, other):
        return self.combine(other)


class CrossMetric(CrossModule, abc.ABC):
    @abc.abstractmethod
    def prepare(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def result(self):
        if self._setup:
            return self.present()

        return None

    def __call__(self, *args, **kwargs):
        return self + self.prepare(*args, **kwargs)


def _sum(left, right):
    return left + right
