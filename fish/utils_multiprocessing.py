from types import FunctionType
import marshal

def _applicable(*args, **kwargs):
  name = kwargs['__pw_name']
  code = marshal.loads(kwargs['__pw_code'])
  gbls = globals() #gbls = marshal.loads(kwargs['__pw_gbls'])
  defs = marshal.loads(kwargs['__pw_defs'])
  clsr = marshal.loads(kwargs['__pw_clsr'])
  fdct = marshal.loads(kwargs['__pw_fdct'])
  func = FunctionType(code, gbls, name, defs, clsr)
  func.fdct = fdct
  del kwargs['__pw_name']
  del kwargs['__pw_code']
  del kwargs['__pw_defs']
  del kwargs['__pw_clsr']
  del kwargs['__pw_fdct']
  return func(*args, **kwargs)

def make_applicable(f, *args, **kwargs):
  if not isinstance(f, FunctionType): raise ValueError('argument must be a function')
  kwargs['__pw_name'] = f.__name__  # edited
  kwargs['__pw_code'] = marshal.dumps(f.__code__)   # edited
  kwargs['__pw_defs'] = marshal.dumps(f.__defaults__)  # edited
  kwargs['__pw_clsr'] = marshal.dumps(f.__closure__)  # edited
  kwargs['__pw_fdct'] = marshal.dumps(f.__dict__)   # edited
  return _applicable, args, kwargs

def _mappable(x):
  x,name,code,defs,clsr,fdct = x
  code = marshal.loads(code)
  gbls = globals() #gbls = marshal.loads(gbls)
  defs = marshal.loads(defs)
  clsr = marshal.loads(clsr)
  fdct = marshal.loads(fdct)
  func = FunctionType(code, gbls, name, defs, clsr)
  func.fdct = fdct
  return func(x)
  