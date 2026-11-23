from collections.abc import Callable
from typing import TypeVar

T = TypeVar('T')

def lazy_evaluation(producer: Callable[[], T]) -> Callable[[], T]:
  value_generated = False
  value = None
  
  def getter():
    nonlocal value_generated, value
    
    if not value_generated:
      value = producer()
      value_generated = True
    
    return value_generated
  
  return getter
