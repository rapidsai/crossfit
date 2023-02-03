from typing import Dict, List
from dataclasses import field, MISSING, Field
from copy import deepcopy


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
    metadata["state"] = True
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
        self.update_state(**kwargs)
        self._setup = True

    def update_state(self, **kwargs):
        for name, state_field in self.field_dict().items():
            if name in kwargs:
                setattr(self, name, kwargs[name])
            else:
                if state_field.default_factory is not MISSING:
                    setattr(self, name, state_field.default_factory())
                else:
                    setattr(self, name, state_field.default)

        return self

    def with_state(self, **kwargs):
        return deepcopy(self).update_state(**kwargs)

    @classmethod
    def field_dict(cls) -> Dict[str, Field]:
        output = {}
        for name in dir(cls):
            if not name.startswith("_"):
                part = getattr(cls, name)
                if isinstance(part, Field):
                    output[name] = part
                    part.name = name

        return output

    @classmethod
    def fields(cls) -> List[Field]:
        return list(cls.field_dict().values())

    @property
    def state_dict(self):
        output = {}
        for f in self.fields():
            output[f.name] = getattr(self, f.name)

        return output

    def combine(self, other):
        merged_fields = {}

        for state_field in self.fields():
            _type = get_origin(state_field.type)
            _type = _type or state_field.type

            if "combine" in state_field.metadata:
                merged_fields[state_field.name] = state_field.metadata["combine"](
                    getattr(self, state_field.name),
                    getattr(other, state_field.name),
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

        return deepcopy(self).update_state(**merged_fields)

    def __add__(self, other):
        return self.combine(other)


def _sum(left, right):
    return left + right
