/**
 * \file pde_model.cpp
 * \brief definition PDEModel methods
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

namespace davinci {
//==============================================================================
template<typename MeshT, typename Equation>
PDEModel<MeshT,Equation>::PDEModel(
    ostream& out, const RCP<const Comm<int> >& comm)
    : Model(out,comm), mesh_(out), work_set_(out) {
}
//==============================================================================
template<typename MeshT, typename Equation>
void PDEModel<MeshT,Equation>::InitializeMesh(ParameterList& p) {
  mesh_.Initialize(p);
}
//==============================================================================
template<typename MeshT, typename Equation>
void PDEModel<MeshT,Equation>::InitializeEquationSet(ParameterList& p) {
  typedef typename MeshT::elem_idx_type_ elem_idx_type;
  elem_idx_type num_elems = mesh_.get_num_elems();
  int batch_size = p.get("Batch Size", num_elems);
  Teuchos::RCP<const CellTopologyData> cell(
      p.get<const CellTopologyData*>("Topology"), false);
  work_set_.DefineTopology(cell);
  work_set_.DefineCubature(p.get<int>("Cubature Degree"));
  typedef typename Equation::RealT ScalarT;
  //work_set_.DefineBasis(*p.get("Basis"));
  //work_set_.resize(num_sets_);
  //ArrayRCP<Equation>::iterator set_it;
  //for (set_it = work_set_.begin(); set_it != work_set.end(); set_it++)
}

} // namespace davinci
