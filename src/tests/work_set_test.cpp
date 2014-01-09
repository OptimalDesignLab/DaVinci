/**
 * \file work_set_test.cpp
 * \brief unit test for the WorkSet<ScalarT> class
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#include <boost/test/unit_test.hpp>
#include <Teuchos_Time.hpp>
#include "Shards_CellTopology.hpp"
#include "Shards_CellTopologyData.h"
#include "Intrepid_HGRAD_HEX_C1_FEM.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "work_set.hpp"
#include "simple_mesh.hpp"

using Intrepid::FieldContainer;
using davinci::WorkSet;
using davinci::SimpleMesh;

BOOST_AUTO_TEST_SUITE(WorkSet_suite)

BOOST_AUTO_TEST_CASE(Constructors) {
  WorkSet<double,SimpleMesh> MyWorkSet(std::cout);  
}

BOOST_AUTO_TEST_CASE(Topology) {
  WorkSet<double,SimpleMesh> MyWorkSet(std::cout);
  Teuchos::RCP<const CellTopologyData> cell(
      shards::getCellTopologyData<shards::Triangle<3> >(), false);
  MyWorkSet.DefineTopology(cell);
  cell = Teuchos::rcp(
      shards::getCellTopologyData<shards::Tetrahedron<4> >(), false);
  MyWorkSet.DefineTopology(cell);
}

BOOST_AUTO_TEST_CASE(Cubature) {
  WorkSet<double,SimpleMesh> MyWorkSet(std::cout);
  Teuchos::RCP<const CellTopologyData> cell(
      shards::getCellTopologyData<shards::Triangle<3> >(), false);
  MyWorkSet.DefineTopology(cell);
  for (int deg = 1; deg < 10; deg++)
    MyWorkSet.DefineCubature(deg);
}

BOOST_AUTO_TEST_CASE(Basis) {
  typedef double ScalarT;
  WorkSet<ScalarT,SimpleMesh> MyWorkSet(std::cout);
  Teuchos::RCP<const CellTopologyData> cell(
      shards::getCellTopologyData<shards::Triangle<3> >(), false);
  MyWorkSet.DefineTopology(cell);
  int degree = 2;
  MyWorkSet.DefineCubature(degree);
  Intrepid::Basis_HGRAD_TRI_C1_FEM<ScalarT, FieldContainer<ScalarT>
                                   > tri_hgrad_basis;
  MyWorkSet.DefineBasis(tri_hgrad_basis);
}

BOOST_AUTO_TEST_CASE(ResizeSets) {
  WorkSet<double,SimpleMesh> MyWorkSet(std::cout);
  Teuchos::RCP<const CellTopologyData> cell(
      shards::getCellTopologyData<shards::Triangle<3> >(), false);
  MyWorkSet.DefineTopology(cell);
  int degree = 2;
  MyWorkSet.DefineCubature(degree);
  int total_elems = 100;
  for (int nelems = 1; nelems <= 100; nelems++)
    MyWorkSet.ResizeSets(total_elems, nelems);
}

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

BOOST_AUTO_TEST_SUITE_END()
