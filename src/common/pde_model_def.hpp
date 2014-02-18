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
void PDEModel<MeshT>::CreateLinearSystemWorkSets(
    ParameterList& options,
    const RCP<WorkSetFactoryBase<MeshT> >& workset_factory) {
  BOOST_ASSERT_MSG(options.isParameter("basis degree"),
                   "[basis degree] must be defined in options");
  options.set("num local elems", mesh_.get_num_elems());
  set_num_pdes(workset_factory->NumPDEs());
  Array<RCP<const BasisT> > bases;
  mesh_.GetIntrepidBases(options.get<int>("basis degree"), bases);
  Array<RCP<const BasisT> >::iterator it;
  for (it = bases.begin(); it != bases.end(); ++it) 
    workset_factory->BuildLinearSystemWorkSet(options, *it, workset_);
#ifdef DAVINCI_VERBOSE
  *out_ << "PDEModel::CreateLinearSystemWorkSets(): "
        << "WorkSet list created/appended to.\n";
#endif
}
//==============================================================================
template<typename MeshT>
void PDEModel<MeshT>::BuildLinearSystem() {
  sol_ = rcp(new VectorT(map_,1));
  sol_->putScalar(1.0);
  rhs_ = rcp(new VectorT(map_,1));
  jac_ = rcp(new MatrixT(jac_graph_));
  jac_->fillComplete();
  typename Array<RCP<WorkSetBase<MeshT> > >::iterator it;
  for (it = workset_.begin(); it != workset_.end(); ++it)
    (*it)->BuildSystem(mesh_, sol_, rhs_, jac_);
#ifdef DAVINCI_VERBOSE
  *out_ << "PDEModel::BuildLinearSystem(): "
        << "system Jacobian and RHS generated.\n";
#endif
}
//==============================================================================
} // namespace davinci
