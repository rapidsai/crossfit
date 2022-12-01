import abc
import functools as ft
import inspect
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any


def static_field(**kwargs):
    """Used for marking that a field should _not_ be treated as a leaf of the PyTree
    of a [`equinox.Module`][]. (And is instead treated as part of the structure, i.e.
    as extra metadata.)
    !!! example
        ```python
        class MyModule(equinox.Module):
            normal_field: int
            static_field: int = equinox.static_field()
        mymodule = MyModule("normal", "static")
        leaves, treedef = jtu.tree_flatten(mymodule)
        assert leaves == ["normal"]
        assert "static" in str(treedef)
        ```
    In practice this should rarely be used; it is usually preferential to just filter
    out each field with `eqx.filter` whenever you need to select only some fields.
    **Arguments:**
    - `**kwargs`: If any are passed then they are passed on to `datacalss.field`.
        (Recall that Equinox uses dataclasses for its modules.)
    """
    try:
        metadata = dict(kwargs["metadata"])
    except KeyError:
        metadata = kwargs["metadata"] = {}
    if "static" in metadata:
        raise ValueError("Cannot use metadata with `static` already set.")
    metadata["static"] = True
    return field(**kwargs)


class _wrap_method:
    def __init__(self, method):
        self.method = method
        if getattr(self.method, "__isabstractmethod__", False):
            self.__isabstractmethod__ = self.method.__isabstractmethod__

    def __get__(self, instance, owner):
        if instance is None:
            return self.method
        # return jtu.Partial(self.method, instance)
        return instance


def _not_magic(k: str) -> bool:
    return not (k.startswith("__") and k.endswith("__"))


# Inherits from abc.ABCMeta as a convenience for a common use-case.
# It's not a feature we use ourselves.
class _ModuleMeta(abc.ABCMeta):
    def __new__(mcs, name, bases, dict_):
        dict_ = {
            # k: _wrap_method(v) if _not_magic(k) and inspect.isfunction(v) else v
            k: v if _not_magic(k) and inspect.isfunction(v) else v
            for k, v in dict_.items()
        }
        cls = super().__new__(mcs, name, bases, dict_)
        # Do override subclasses' dataclass-__init__-s. (None of which call super, so
        # they must be overriden.)
        # Don't override custom __init__'s, which leads to poor ergonomics:
        # e.g. if `B` has a custom init then `class A(B): pass` would otherwise set a
        # dataclass init that overrides the custom __init__.
        _init = cls._has_dataclass_init = _has_dataclass_init(cls)
        if _init:
            init_doc = cls.__init__.__doc__
        if not is_dataclass(cls):
            cls = dataclass(eq=False, repr=False, frozen=True, init=_init)(cls)
        if _init:
            cls.__init__.__doc__ = init_doc
        return cls

    def __call__(cls, *args, **kwargs):
        self = cls.__new__(cls, *args, **kwargs)
        # Defreeze it during __init__
        initable_cls = _make_initable(cls, wraps=False)
        object.__setattr__(self, "__class__", initable_cls)
        try:
            cls.__init__(self, *args, **kwargs)
        finally:
            object.__setattr__(self, "__class__", cls)

        missing_names = {
            field.name
            for field in fields(cls)
            if field.init and field.name not in dir(self)
        }
        if len(missing_names):
            raise ValueError(
                f"The following fields were not initialised during __init__: {missing_names}"
            )
        return self


@ft.lru_cache(maxsize=128)
def _make_initable(cls: _ModuleMeta, wraps: bool) -> _ModuleMeta:
    if wraps:
        field_names = {
            "__module__",
            "__name__",
            "__qualname__",
            "__doc__",
            "__annotations__",
            "__wrapped__",
        }
    else:
        field_names = {field.name for field in fields(cls)}

    class _InitableModule(cls):
        pass

    # Done like this to avoid dataclasses complaining about overriding setattr on a
    # frozen class.
    def __setattr__(self, name, value):
        if name in field_names:
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(f"Cannot set attribute {name}")

    _InitableModule.__setattr__ = __setattr__

    return _InitableModule


def _has_dataclass_init(cls: _ModuleMeta) -> bool:
    if "__init__" in cls.__dict__:
        return False
    return cls._has_dataclass_init


class Module(metaclass=_ModuleMeta):
    """Base class. Create your model by inheriting from this.
    **Fields**
    Specify all its fields at the class level (identical to
    [dataclasses](https://docs.python.org/3/library/dataclasses.html)). This defines
    its children as a PyTree.
    ```python
    class MyModule(equinox.Module):
        weight: jax.numpy.ndarray
        bias: jax.numpy.ndarray
        submodule: equinox.Module
    ```
    **Initialisation**
    A default `__init__` is automatically provided, which just fills in fields with the
    arguments passed. For example `MyModule(weight, bias, submodule)`.
    Alternatively (quite commonly) you can provide an `__init__` method yourself:
    ```python
    class MyModule(equinox.Module):
        weight: jax.numpy.ndarray
        bias: jax.numpy.ndarray
        submodule: equinox.Module
        def __init__(self, in_size, out_size, key):
            wkey, bkey, skey = jax.random.split(key, 3)
            self.weight = jax.random.normal(wkey, (out_size, in_size))
            self.bias = jax.random.normal(bkey, (out_size,))
            self.submodule = equinox.nn.Linear(in_size, out_size, key=skey)
    ```
    **Methods**
    It is common to create some methods on the class -- for example to define the forward pass of a model.
    ```python
    class MyModule(equinox.Module):
        ...  # as above
        def __call__(self, x):
            return self.submodule(x) + self.weight @ x + self.bias
    ```
    !!! tip
        You don't have to define `__call__`:
        - You can define other methods if you want.
        - You can define multiple methods if you want.
        - You can define no methods if you want. (And just use `equinox.Module` as a
            nice syntax for custom PyTrees.)
        No method is special-cased.
    **Usage**
    After you have defined your model, then you can use it just like any other PyTree
    -- that just happens to have some methods attached. In particular you can pass it
    around across `jax.jit`, `jax.grad` etc. in exactly the way that you're used to.
    !!! example
        If you wanted to, then it would be completely safe to do
        ```python
        class MyModule(equinox.Module):
            ...
            @jax.jit
            def __call__(self, x):
                ...
        ```
        because `self` is just a PyTree. Unlike most other neural network libaries, you
        can mix Equinox and native JAX without any difficulties at all.
    """

    _has_dataclass_init = True

    def __hash__(self):
        return hash(tuple(jtu.tree_leaves(self)))

    def __eq__(self, other):
        return tree_equal(self, other)

    def __repr__(self):
        state = self.state()
        if len(state) == 1:
            state_str = state[0]
        else:
            state_str = ", ".join(f"{k}={v}" for k, v in self.state_dict().items())
        return f"{self.__class__.__name__}({state_str})"
        
    def fields(self, static=False):
        if static:
            for field in fields(self):
                if field.metadata.get("static", False):
                    yield field
        else:
            for field in fields(self):
                if not field.metadata.get("static", False):
                    yield field
        
    def state(self, flattened=True):    
        return tuple(getattr(self, field.name) for field in self.fields())
            
    def state_dict(self):
        output = {}
        for field in self.fields():
            value = getattr(self, field.name)
            if isinstance(value, Module):                
                for name, val in value.state_dict().items():
                    output[".".join([field.name, name])] = val
            else:
                output[field.name] = value
        
        return output

    @classmethod
    def tree_unflatten(cls, aux, dynamic_field_values):
        self = cls.__new__(cls)
        dynamic_field_names, static_field_names, static_field_values = aux
        for name, value in zip(dynamic_field_names, dynamic_field_values):
            object.__setattr__(self, name, value)
        for name, value in zip(static_field_names, static_field_values):
            object.__setattr__(self, name, value)
        return self


# Modifies in-place, just like functools.update_wrapper
def module_update_wrapper(wrapper: Module, wrapped) -> Module:
    cls = wrapper.__class__
    initable_cls = _make_initable(cls, wraps=True)
    object.__setattr__(wrapper, "__class__", initable_cls)
    try:
        # updated = ("__dict__",) is the default, but that's a bit much.
        # It's common/possible for wrapper and wrapped to both be classes
        # implementing __call__, in which case copying __dict__ over basically
        # just breaks the wrapper class.
        ft.update_wrapper(wrapper, wrapped, updated=())
    finally:
        object.__setattr__(wrapper, "__class__", cls)
    return wrapper


class Static(Module):
    value: Any = static_field()
    
    
if __name__ == '__main__':
    
    @dataclass(frozen=True)
    class MyModule(Module):
        normal_field: int
        static_field: int = static_field()
        
        
    class Nested(Module):
        submodule: MyModule
        
        
    class Complex(Nested):
        field: float
        
        
    class Custom(Module):
        field: float
        
        def __init__(self, field: float):
            self.field = field
    
    
    simple = MyModule("normal", "static")
    # nested = Complex(simple, 1.0)
    # state = simple.state_dict()
    
    
    custom = Custom(1.0)
    custom.state_dict()
    
    Custom(1.0)
    
    
    
    a = 5