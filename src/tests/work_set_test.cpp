/**
 * \file work_set_test.cpp
 * \brief unit test for the WorkSet<NodeT,ScalarT,MeshT> class
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#include <boost/test/unit_test.hpp>
#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_Time.hpp"
#include "Shards_CellTopology.hpp"
#include "Shards_CellTopologyData.h"
#include "Intrepid_HGRAD_HEX_C1_FEM.hpp"
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
using Teuchos::ParameterList;
using Teuchos::Comm;
using Teuchos::DefaultComm;
using Intrepid::FieldContainer;
using davinci::WorkSet;
using davinci::SimpleMesh;
using davinci::Evaluator;
using davinci::MetricJacobian;
using davinci::Laplace;

typedef Tpetra::BlockMultiVector<double,int,int> Vector;
typedef Tpetra::VbrMatrix<double,int,int> Matrix;
typedef Tpetra::BlockMap<int,int> Map;
//typedef Tpetra::BlockMap<int,int> BlockMap;

BOOST_AUTO_TEST_SUITE(WorkSet_suite)

BOOST_AUTO_TEST_CASE(Constructors) {
  WorkSet<double,double,SimpleMesh,Vector,Matrix> MyWorkSet(std::cout);
}

BOOST_AUTO_TEST_CASE(Topology) {
  WorkSet<double,double,SimpleMesh,Vector,Matrix> MyWorkSet(std::cout);
  Teuchos::RCP<const CellTopologyData> cell(
      shards::getCellTopologyData<shards::Triangle<3> >(), false);
  MyWorkSet.DefineTopology(cell);
  cell = Teuchos::rcp(
      shards::getCellTopologyData<shards::Tetrahedron<4> >(), false);
  MyWorkSet.DefineTopology(cell);
}

BOOST_AUTO_TEST_CASE(Cubature) {
  WorkSet<double,double,SimpleMesh,Vector,Matrix> MyWorkSet(std::cout);
  Teuchos::RCP<const CellTopologyData> cell(
      shards::getCellTopologyData<shards::Triangle<3> >(), false);
  MyWorkSet.DefineTopology(cell);
  for (int deg = 1; deg < 10; deg++)
    MyWorkSet.DefineCubature(deg);
}

BOOST_AUTO_TEST_CASE(Basis) {
  typedef double ScalarT;
  WorkSet<ScalarT,ScalarT,SimpleMesh,Vector,Matrix> MyWorkSet(std::cout);
  Teuchos::RCP<const CellTopologyData> cell(
      shards::getCellTopologyData<shards::Triangle<3> >(), false);
  MyWorkSet.DefineTopology(cell);
  int degree = 2;
  MyWorkSet.DefineCubature(degree);
  Intrepid::Basis_HGRAD_TRI_C1_FEM<ScalarT, FieldContainer<ScalarT>
                                   > tri_hgrad_basis;
  MyWorkSet.DefineBasis(tri_hgrad_basis);
}

BOOST_AUTO_TEST_CASE(Evaluators) {
  typedef double ScalarT;
  typedef double NodeT;
  WorkSet<NodeT,ScalarT,SimpleMesh,Vector,Matrix> MyWorkSet(std::cout);
  std::list<Evaluator<NodeT,ScalarT>* > evaluators;
  evaluators.push_back(new MetricJacobian<NodeT,ScalarT>());
  evaluators.push_back(new Laplace<NodeT,ScalarT>());
  MyWorkSet.DefineEvaluators(evaluators);
}

BOOST_AUTO_TEST_CASE(ResizeSets) {  
  WorkSet<double,double,SimpleMesh,Vector,Matrix> MyWorkSet(std::cout);
  Teuchos::RCP<const CellTopologyData> cell(
      shards::getCellTopologyData<shards::Triangle<3> >(), false);
  MyWorkSet.DefineTopology(cell);
  int degree = 2;
  MyWorkSet.DefineCubature(degree);
  Intrepid::Basis_HGRAD_TRI_C1_FEM<double, FieldContainer<double>
                                   > tri_hgrad_basis;
  MyWorkSet.DefineBasis(tri_hgrad_basis);  
  std::list<Evaluator<double,double>* > evaluators;
  evaluators.push_back(new MetricJacobian<double,double>());
  evaluators.push_back(new Laplace<double,double>());
  MyWorkSet.DefineEvaluators(evaluators);
  int num_pdes = 1;
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
  //int Nx = 10, Ny = 10;
  int Nx = 2, Ny = 2;
  Mesh.BuildRectangularMesh(Lx, Ly, Nx, Ny);

  // Use mesh to create Tpetra map and graph
  RCP<const Tpetra::BlockMap<SimpleMesh::LocIdxT,SimpleMesh::GlbIdxT> > map;
  const int num_pdes = 1;
  Mesh.BuildTpetraMap(num_pdes, map);
  RCP<Tpetra::BlockCrsGraph<int,int> > jac_graph;
  Mesh.BuildMatrixGraph(map, jac_graph);
  
  // Create a Workset for the Laplace PDE
  typedef Sacado::Fad::SFad<double,3*num_pdes> ADType;
  WorkSet<double,ADType,SimpleMesh,Vector,Matrix> MyWorkSet(out);
  Teuchos::RCP<const CellTopologyData> cell(
      shards::getCellTopologyData<shards::Triangle<3> >(), false);
  MyWorkSet.DefineTopology(cell);
  int degree = 2;
  MyWorkSet.DefineCubature(degree);
  Intrepid::Basis_HGRAD_TRI_C1_FEM<double, FieldContainer<double>
                                   > tri_hgrad_basis;
  MyWorkSet.DefineBasis(tri_hgrad_basis);
  std::list<Evaluator<double,ADType>* > evaluators;
  evaluators.push_back(new MetricJacobian<double,ADType>());
  evaluators.push_back(new Laplace<double,ADType>());
  MyWorkSet.DefineEvaluators(evaluators);
  int total_elems = Mesh.get_num_elems();
  //int nelems = 10;
  int nelems = 2;
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

#if 0
BOOST_AUTO_TEST_CASE(CopyMeshCoords) {
  WorkSet<double,SimpleMesh> MyWorkSet(std::cout);
  Teuchos::RCP<const CellTopologyData> cell(
      shards::getCellTopologyData<shards::Triangle<3> >(), false);
  MyWorkSet.DefineTopology(cell);
  int degree = 2;
  MyWorkSet.DefineCubature(degree);

  SimpleMesh Mesh(std::cout);
  double Lx = 1.0, Ly = 1.0;
  int Nx = 1000, Ny = 1000;
  Mesh.BuildRectangularMesh(Lx, Ly, Nx, Ny);

  int batch_size = 9;
  MyWorkSet.ResizeSets(Mesh.get_num_elems(), batch_size);
  std::div_t div_result = std::div(Mesh.get_num_elems()-1, batch_size);
  int num_batch = div_result.quot+1;
  std::cout << "batch_size = " << batch_size << "\n";
  std::cout << "num_batch = " << num_batch << "\n";
  std::cout << "remainder = " << div_result.rem+1 << "\n";
  for (int bi = 0; bi < num_batch; bi++)
    MyWorkSet.CopyMeshCoords(Mesh, bi);
  
  // Teuchos::Time workset_time("Time to access inside WorkSet: ");
  // Teuchos::Time direct_time("Time to access directly      : ");
  // workset_time.start();
  // MyWorkSet.BuildSystem(Mesh);
  // workset_time.stop();
  // direct_time.start();
  // for (int ielem = 0; ielem < Mesh.get_num_elems(); ielem++)
  //   for (int i = 0; i < 3; i++)
  //     int tmp = Mesh.ElemToNode(ielem, i);
  // direct_time.stop();
  // std::cout << workset_time.name() << workset_time.totalElapsedTime() << "\n";
  // std::cout << direct_time.name() << direct_time.totalElapsedTime() << "\n";
}
#endif

BOOST_AUTO_TEST_SUITE_END()
