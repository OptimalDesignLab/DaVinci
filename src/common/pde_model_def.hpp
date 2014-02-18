/**
 * \file pde_model.cpp
 * \brief definition PDEModel methods
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

namespace davinci {
//==============================================================================
template<typename MeshT>
PDEModel<MeshT>::PDEModel(
    ostream& out, const RCP<const Comm<int> >& comm)
    : Model(out,comm), mesh_(out,comm) {
  out_ = Teuchos::rcp(&out, false);
  num_pdes_ = -1;
#ifdef DAVINCI_VERBOSE
  *out_ << "PDEModel::PDEModel(): constructed PDEModel object.\n";
#endif
}
//==============================================================================
template<typename MeshT>
void PDEModel<MeshT>::set_num_pdes(const int& num_pdes) {
  BOOST_ASSERT_MSG(num_pdes > 0, "number of PDEs must be positive");
  num_pdes_ = num_pdes;
#ifdef DAVINCI_VERBOSE
  *out_ << "PDEModel::set_num_pdes(): number of PDEs set to " << num_pdes_
        << "\n";
#endif
}
//==============================================================================
template<typename MeshT>
void PDEModel<MeshT>::InitializeMesh(ParameterList& p) {
  mesh_.Initialize(p);
}
//==============================================================================
template<typename MeshT>
void PDEModel<MeshT>::CreateMapAndJacobianGraph() {
  BOOST_ASSERT_MSG(num_pdes_ > 0, "number of PDEs must be defined");
  mesh_.BuildTpetraMap(num_pdes_, map_);
  mesh_.BuildMatrixGraph(map_, jac_graph_);
#ifdef DAVINCI_VERBOSE
  *out_ << "PDEModel::CreateMapAndJacobianGraph(): "
        << "Tpetra map and Jacobian graph generated.\n";
#endif
}
//==============================================================================
template<typename MeshT>
void PDEModel<MeshT>::BuildLinearSystemWorkSets(
    const RCP<WorkSetFactoryBase<MeshT> >& workset_factory,
    const int& degree) {
  BOOST_ASSERT_MSG(degree > 0, "polynomial degree must be positive");
  Array<RCP<const BasisT> > bases;
  mesh_.GetIntrepidBases(degree, bases);
  Array<RCP<const BasisT> >::iterator it;
  for (it = bases.begin(); it != bases.end(); ++it) 
    workset_factory->BuildLinearSystemWorkSet(*it, workset_);
#ifdef DAVINCI_VERBOSE
  *out_ << "PDEModel::BuildLinearSystemWorkSets(): "
        << "WorkSet list created/appended to.\n";
#endif
}
//==============================================================================
#if 0
template<typename MeshT>
void PDEModel<MeshT>::InitializeEquationSet(ParameterList& p) {
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
#endif
} // namespace davinci
