package grad

import (
	"gograd/tensor/types"
	"sync"
)

type VarSet[T types.TensorType] struct {
	values map[*Var[T]]struct{}
	lock   sync.RWMutex
}

func CreateVarSet[T types.TensorType](vars ...*Var[T]) *VarSet[T] {
	set := &VarSet[T]{
		values: make(map[*Var[T]]struct{}),
	}
	if len(vars) > 0 {
		for _, v := range vars {
			set.Add(v)
		}
	}
	return set
}

// basics
func (set *VarSet[T]) Add(value *Var[T]) {
	set.lock.Lock()
	defer set.lock.Unlock()
	set.values[value] = struct{}{}
}

func (set *VarSet[T]) Contains(value *Var[T]) bool {
	set.lock.RLock()
	defer set.lock.RUnlock()
	_, ok := set.values[value]
	return ok
}

func (set *VarSet[T]) Remove(value *Var[T]) {
	set.lock.Lock()
	defer set.lock.Unlock()
	delete(set.values, value)
}

func (set *VarSet[T]) ToList() []*Var[T] {
	out := make([]*Var[T], len(set.values))
	i := 0
	for v := range set.values {
		out[i] = v
		i++
	}
	return out
}
