from dataclasses import dataclass

import crossfit as sf


@dataclass(frozen=True)
class MyModule(sf.Module):
    normal_field: int
    static_field: int = sf.static_field()
    
    
class Nested(sf.Module):
    submodule: MyModule
    
    
class Complex(Nested):
    field: float
    
    
class Custom(sf.Module):
    field: float
    
    def __init__(self, field: float):        
        super().__init__()
        self.field = field
        
        
def test_simple_module():
    simple = MyModule(1, 2)
    assert isinstance(simple, sf.Module)
    
    
def test_with_custom_init():
    custom = Custom(1.0)
    assert isinstance(custom, sf.Module)
