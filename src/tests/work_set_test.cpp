/**
 * \file work_set_test.cpp
 * \brief unit test for the WorkSet<NodeT,ScalarT,MeshT,BasisT> class
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#include <boost/test/unit_test.hpp>
#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_Time.hpp"
#include "Shards_CellTopology.hpp"
#include "Shards_CellTopologyData.h"
//#include "Intrepid_HGRAD_HEX_C1_FEM.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Tpetra_DefaultPlatform.hpp"
#include "Tpetra_BlockMultiVector_decl.hpp"
#include "Tpetra_VbrMatrix.hpp"
#include "Sacado_Fad_SFad.hpp"

#include "work_set.hpp"
#include "simple_mesh.hpp"
#include "metric_jacobian.hpp"
#include "laplace.hpp"

using Teuchos::GlobalMPISession;
using Teuchos::RCP;
using Teuchos::Array;
using Teuchos::ParameterList;
using Teuchos::Comm;
using Teuchos::DefaultComm;
using Intrepid::Basis;
using Intrepid::FieldContainer;
using davinci::WorkSet;
using davinci::SimpleMesh;
using davinci::Evaluator;
using davinci::MetricJacobian;
using davinci::Laplace;

typedef Intrepid::Basis_HGRAD_TRI_C1_FEM<double, FieldContainer<double> >
TriBasis;
typedef RCP<Basis<double, FieldContainer<double> > > BasisRCP;
typedef Tpetra::BlockMultiVector<double,int,int> Vector;
typedef Tpetra::VbrMatrix<double,int,int> Matrix;
typedef Tpetra::BlockMap<int,int> Map;
const int num_pdes = 1;
typedef Sacado::Fad::SFad<double,3*num_pdes> ADType;
//typedef Tpetra::BlockMap<int,int> BlockMap;

BOOST_AUTO_TEST_SUITE(WorkSet_suite)

BOOST_AUTO_TEST_CASE(Constructors) {
  BasisRCP basis = Teuchos::rcp(new TriBasis());
  WorkSet<double,ADType,SimpleMesh> MyWorkSet(basis, std::cout);
}

BOOST_AUTO_TEST_CASE(Cubature) {
  BasisRCP basis = Teuchos::rcp(new TriBasis());
  WorkSet<double,ADType,SimpleMesh> MyWorkSet(basis, std::cout);
  for (int deg = 1; deg < 10; deg++)
    MyWorkSet.DefineCubature(deg);
}

BOOST_AUTO_TEST_CASE(EvaluateBasis) {
  BasisRCP basis = Teuchos::rcp(new TriBasis());
  WorkSet<double,ADType,SimpleMesh> MyWorkSet(basis, std::cout);
  int deg = 2;
  MyWorkSet.DefineCubature(deg);
  MyWorkSet.EvaluateBasis();
}

BOOST_AUTO_TEST_CASE(Evaluators) {
  typedef ADType ScalarT;
  typedef double NodeT;
  BasisRCP basis = Teuchos::rcp(new TriBasis());
  WorkSet<NodeT,ScalarT,SimpleMesh> MyWorkSet(basis, std::cout);
  Array<RCP<Evaluator<NodeT,ScalarT> > > evaluators;
  evaluators.push_back(Teuchos::rcp(new MetricJacobian<NodeT,ScalarT>()));
  evaluators.push_back(Teuchos::rcp(new Laplace<NodeT,ScalarT>()));
  MyWorkSet.DefineEvaluators(evaluators);
}

BOOST_AUTO_TEST_CASE(Constructor_with_Evaluators) {
  typedef ADType ScalarT;
  typedef double NodeT;
  Array<RCP<Evaluator<NodeT,ScalarT> > > evaluators;
  evaluators.push_back(Teuchos::rcp(new MetricJacobian<NodeT,ScalarT>()));
  evaluators.push_back(Teuchos::rcp(new Laplace<NodeT,ScalarT>()));
  BasisRCP basis = Teuchos::rcp(new TriBasis());
  const int num_pdes = 1;
  WorkSet<NodeT,ScalarT,SimpleMesh> MyWorkSet(basis, evaluators, num_pdes,
                                              std::cout);
}

BOOST_AUTO_TEST_CASE(ResizeSets) {
  BasisRCP basis = Teuchos::rcp(new TriBasis());
  WorkSet<double,ADType,SimpleMesh> MyWorkSet(basis, std::cout);
  int degree = 2;
  MyWorkSet.DefineCubature(degree);
  MyWorkSet.EvaluateBasis();

  Array<RCP<Evaluator<double,ADType> > > evaluators;
  evaluators.push_back(Teuchos::rcp(new MetricJacobian<double,ADType>()));
  evaluators.push_back(Teuchos::rcp(new Laplace<double,ADType>()));
  MyWorkSet.DefineEvaluators(evaluators);
  int total_elems = 100;
  for (int nelems = 1; nelems <= total_elems; nelems++)
    MyWorkSet.ResizeSets(num_pdes, total_elems, nelems);
}

BOOST_AUTO_TEST_CASE(BuildSystem) {
  // create Teuchos communicator
  GlobalMPISession(&boost::unit_test::framework::master_test_suite().argc,
                   &boost::unit_test::framework::master_test_suite().argv,
                   NULL);
  RCP<const Comm<int> > comm =
      Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
  //DefaultComm<int>::getComm();

  // stream for output
  const int myRank = comm->getRank();
  const int numProcs = comm->getSize();
  Teuchos::oblackholestream blackHole;
  std::ostream& out = (comm->getRank() == 0) ? std::cout : blackHole;
  
  // Define a rectangular mesh  
  SimpleMesh Mesh(out, comm);
  double Lx = 1.0, Ly = 1.0;
  int Nx = 10, Ny = 10;
  //int Nx = 2, Ny = 2;
  Mesh.BuildRectangularMesh(Lx, Ly, Nx, Ny);

  // Use mesh to create Tpetra map and graph
  RCP<const Tpetra::BlockMap<SimpleMesh::LocIdxT,SimpleMesh::GlbIdxT> > map;
  Mesh.BuildTpetraMap(num_pdes, map);
  RCP<Tpetra::BlockCrsGraph<int,int> > jac_graph;
  Mesh.BuildMatrixGraph(map, jac_graph);
  
  // Create a Workset for the Laplace PDE
  BasisRCP basis = Teuchos::rcp(new TriBasis());
  WorkSet<double,ADType,SimpleMesh> MyWorkSet(basis, out);
  int degree = 2;
  MyWorkSet.DefineCubature(degree);
  MyWorkSet.EvaluateBasis();
  Array<RCP<Evaluator<double,ADType> > > evaluators;
  evaluators.push_back(Teuchos::rcp(new MetricJacobian<double,ADType>()));
  evaluators.push_back(Teuchos::rcp(new Laplace<double,ADType>()));
  MyWorkSet.DefineEvaluators(evaluators);
  int total_elems = Mesh.get_num_elems();
  int nelems = 10;
  //int nelems = 2;
  MyWorkSet.ResizeSets(num_pdes, total_elems, nelems);
  
  // Create solution, rhs, and jacobian
  RCP<Vector> sol = rcp(
      new Tpetra::BlockMultiVector<double,SimpleMesh::LocIdxT,
      SimpleMesh::GlbIdxT>(map,1));
  sol->putScalar(1.0);
  RCP<Vector> rhs = rcp(
      new Tpetra::BlockMultiVector<double,SimpleMesh::LocIdxT,
      SimpleMesh::GlbIdxT>(map,1));
  RCP<Matrix> jacobian = rcp(
      new Tpetra::VbrMatrix<double,int,int>(jac_graph));
  jacobian->fillComplete();
  MyWorkSet.BuildSystem(Mesh, sol, rhs, jacobian);

  // The Laplacian is applied to a constant, so the residual here should be zero
  // everywhere (no boundary conditions)
  Teuchos::ArrayRCP<const double> rhs_view = rhs->get1dView();
  for (int i = 0; i < rhs->getLocalLength(); i++)
    BOOST_CHECK_SMALL(rhs_view[i], 1e-13);
  
#if 0
  // uncomment to inspect rhs and jacobian
  out << "rhs = ";
  for (int i = 0; i < rhs->getLocalLength(); i++)
    out << rhs_view[i] << " ";
  out << "\n";
  out << "jacobian = ";
  Teuchos::ArrayView<const int> block_cols;
  Teuchos::Array<int> col_stride;
  Teuchos::ArrayRCP<const double> block_entries;
  for (int i = 0; i < map->getNodeNumBlocks(); i++) {
    int row_stride;
    jacobian->getLocalBlockRowView(i, row_stride, block_cols, col_stride,
                                   block_entries);
    out << "i = " << i << ":\n";
    out << "\trow_stride = " << row_stride << "\n";
    out << "\tblock_cols = ";
    for (int j = 0; j < block_cols.size(); j++)
      out << block_cols[j] << " ";
    out << "\n\tcol_stride = ";
    for (int j = 0; j < col_stride.size(); j++)
      out << col_stride[j] << " ";
    out << "\n\tblock_entries = ";
    for (int j = 0; j < block_entries.size(); j++)
      out << block_entries[j] << " ";
    out << "\n";
  }
#endif
}

BOOST_AUTO_TEST_SUITE_END()
