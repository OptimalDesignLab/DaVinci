/**
 * \file pde_model.cpp
 * \brief definition PDEModel methods
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

namespace davinci {
//==============================================================================
template<typename MeshType, typename Equation>
PDEModel<MeshType,Equation>::PDEModel(
    ostream& out, const RCP<const Comm<int> >& comm)
    : Model(out,comm), mesh_(out), work_set_(out) {
}
//==============================================================================
template<typename MeshType, typename Equation>
void PDEModel<MeshType,Equation>::InitializeMesh(ParameterList& p) {
  mesh_.Initialize(p);
}
//==============================================================================
template<typename MeshType, typename Equation>
void PDEModel<MeshType,Equation>::InitializeEquationSet(ParameterList& p) {
  typedef typename MeshType::elem_idx_type_ elem_idx_type;
  elem_idx_type num_elems = mesh_.get_num_elems();
  int batch_size = p.get("Batch Size", num_elems);
  num_sets_ = num_elems/batch_size;
  //work_set_.resize(num_sets_);
  //ArrayRCP<Equation>::iterator set_it;
  //for (set_it = work_set_.begin(); set_it != work_set.end(); set_it++)
}

} // namespace davinci
