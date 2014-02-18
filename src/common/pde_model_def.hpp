/**
 * \file pde_model.cpp
 * \brief definition PDEModel methods
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#include <BelosTpetraAdapter.hpp>
#include <BelosSolverFactory.hpp>

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
  //  Teuchos::rcp_dynamic_cast<MatrixT>(jac_)->fillComplete();
  typename Array<RCP<WorkSetBase<MeshT> > >::iterator it;
  for (it = workset_.begin(); it != workset_.end(); ++it)
    (*it)->BuildSystem(mesh_, sol_, rhs_, jac_);
#ifdef DAVINCI_VERBOSE
  *out_ << "PDEModel::BuildLinearSystem(): "
        << "system Jacobian and RHS generated.\n";
#endif
}
//==============================================================================
template<typename MeshT>
void PDEModel<MeshT>::Solve() {
  // This is a draft version that assumes a linear problem taken from the
  // Trilinos multi-core tutorial

  // Set parameters for Belos solver
  RCP<ParameterList> solverParams = Teuchos::parameterList();
  
  // Set some GMRES parameters.
  //
  // "Num Blocks" = Maximum number of Krylov vectors to store.  This
  // is also the restart length.  "Block" here refers to the ability
  // of this particular solver (and many other Belos solvers) to solve
  // multiple linear systems at a time, even though we may only be
  // solving one linear system in this example.
  //
  // "Maximum Iterations": Maximum total number of iterations,
  // including restarts.
  //
  // "Convergence Tolerance": By default, this is the relative
  // residual 2-norm, although you can change the meaning of the
  // convergence tolerance using other parameters.
  solverParams->set ("Num Blocks", 40);
  solverParams->set ("Maximum Iterations", 400);
  solverParams->set ("Convergence Tolerance", 1.0e-8);

  // Create the GMRES solver using a "factory" and 
  // the list of solver parameters created above.
  Belos::SolverFactory<double, BaseVectorT, BaseMatrixT> factory;
  RCP<Belos::SolverManager<double, BaseVectorT, BaseMatrixT> > solver = 
    factory.create("GMRES", solverParams);

  // Create a LinearProblem struct with the problem to solve.
  // A, X, B, and M are passed by (smart) pointer, not copied.
  RCP<BaseMatrixT> jac_base = Teuchos::rcp_implicit_cast<BaseMatrixT>(jac_);
  RCP<BaseVectorT> sol_base = Teuchos::rcp_implicit_cast<BaseVectorT>(sol_);
  RCP<BaseVectorT> rhs_base = Teuchos::rcp_implicit_cast<BaseVectorT>(rhs_);  
  typedef Belos::LinearProblem<double, BaseVectorT, BaseMatrixT> problem_type;
  RCP<problem_type> problem = rcp(
      new problem_type(jac_base, sol_base, rhs_base));
  // You don't have to call this if you don't have a preconditioner.
  // If prec is null, then Belos won't use a (right) preconditioner.
  RCP<BaseMatrixT> prec;
  problem->setRightPrec(prec);
  // Tell the LinearProblem to make itself ready to solve.
  problem->setProblem();
  // Tell the solver what problem you want to solve.
  solver->setProblem(problem);

  // Attempt to solve the linear system.  result == Belos::Converged 
  // means that it was solved to the desired tolerance.  This call 
  // overwrites sol_ with the computed approximate solution.
  Belos::ReturnType result = solver->solve();

  // Ask the solver how many iterations the last solve() took.
  const int numIters = solver->getNumIters();

  if (result == Belos::Converged) {
    *out_ << "The Belos solve took " << numIters << " iteration(s) to reach "
        "a relative residual tolerance of " << 1.0e-8 << "." << std::endl;
  } else {
    *out_ << "The Belos solve took " << numIters << " iteration(s), "
          << "but did not reach a relative residual tolerance of " << 1.0e-8
          << "." << std::endl;
  }
}
//==============================================================================
} // namespace davinci
