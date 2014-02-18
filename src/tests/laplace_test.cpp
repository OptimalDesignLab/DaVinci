/**
 * \file laplace_test.cpp
 * \brief unit test for the Laplace and LaplaceFactory classes
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#include <boost/test/unit_test.hpp>
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"
#include "Shards_CellTopology.hpp"
#include "Shards_CellTopologyData.h"
#include "Intrepid_Basis.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "metric_jacobian.hpp"
#include "laplace.hpp"
#include "work_set.hpp"
#include "simple_mesh.hpp"

using Teuchos::RCP;
using Intrepid::CellTools;
using Intrepid::Basis;
using Intrepid::FieldContainer;
using davinci::Evaluator;
using davinci::MetricJacobian;
using davinci::Laplace;
using davinci::SimpleMesh;
using davinci::WorkSetBase;
using davinci::WorkSetFactoryBase;
using davinci::LaplaceFactory;
using davinci::WorkSetFactory;

BOOST_AUTO_TEST_SUITE(Laplace_suite)

BOOST_AUTO_TEST_CASE(Constructors) {
  Laplace<double,double> laplace_pde();
}

BOOST_AUTO_TEST_CASE(SetDimensions) {
  Evaluator<double,double>* laplace_pde = new Laplace<double,double>();
  int num_elems = 10;
  int num_nodes_per_elem = 3;
  int num_cub_points = 10;
  int num_ref_basis = 3;
  int dim = 3;
  laplace_pde->SetDimensions(num_elems, num_nodes_per_elem, num_cub_points,
                             num_ref_basis, dim);
}

BOOST_AUTO_TEST_CASE(MemoryRequired) {
  Evaluator<double,double>* laplace_pde = new Laplace<double,double>();
  int num_elems = 10;
  int num_nodes_per_elem = 3;
  int num_cub_points = 10;
  int num_ref_basis = 3;
  int dim = 2;
  laplace_pde->SetDimensions(num_elems, num_nodes_per_elem, num_cub_points,
                             num_ref_basis, dim);
  int mesh_offset = 0;
  int soln_offset = 0;
  int resid_offset = 0;
  std::map<std::string,int> mesh_map_offset, soln_map_offset, resid_map_offset;
  laplace_pde->MemoryRequired(mesh_offset, mesh_map_offset,
                              soln_offset, soln_map_offset,
                              resid_offset, resid_map_offset);
  BOOST_CHECK_EQUAL( mesh_map_offset["jacob_cub"], 0);
  BOOST_CHECK_EQUAL( mesh_map_offset["grads_transformed"],
                     num_elems*num_cub_points);
  BOOST_CHECK_EQUAL( mesh_map_offset["grads_transformed_weighted"],
                     num_elems*num_cub_points*(1 + num_ref_basis*dim));
  BOOST_CHECK_EQUAL( mesh_offset,
                     num_elems*num_cub_points*(1 + 2*num_ref_basis*dim));
  BOOST_CHECK_EQUAL( resid_map_offset["solution_grad"], 0);
  BOOST_CHECK_EQUAL( resid_map_offset["residual"],
                     num_elems*num_cub_points*dim);
  BOOST_CHECK_EQUAL( resid_offset,
                     num_elems*(num_cub_points*dim + num_ref_basis));
  BOOST_CHECK_EQUAL( soln_offset, 0);
}

BOOST_AUTO_TEST_CASE(SetDataView) {
  Evaluator<double,double>* laplace_pde = new Laplace<double,double>();
  int num_elems = 10;
  int num_nodes_per_elem = 3;
  int num_cub_points = 10;
  int num_ref_basis = 3;
  int dim = 2;
  laplace_pde->SetDimensions(num_elems, num_nodes_per_elem, num_cub_points,
                             num_ref_basis, dim);
  int mesh_offset = num_elems*num_cub_points*(1 + dim*dim);
  int soln_offset = num_elems*num_ref_basis;
  int resid_offset = 0;
  std::map<std::string,int> mesh_map_offset, soln_map_offset, resid_map_offset;
  mesh_map_offset["jacob_inv"] = 0;
  mesh_map_offset["jacob_det"] = num_elems*num_cub_points*dim*dim;
  soln_map_offset["solution_coeff"] = 0;
  laplace_pde->MemoryRequired(mesh_offset, mesh_map_offset,
                              soln_offset, soln_map_offset,
                              resid_offset, resid_map_offset);
  Teuchos::ArrayRCP<double> mesh_data(mesh_offset, 1.0);
  Teuchos::ArrayRCP<double> soln_data(soln_offset, 1.0);
  Teuchos::ArrayRCP<typename davinci::Evaluator<double,double>::ResidT>
      resid_data(resid_offset, 1.0);
  laplace_pde->SetDataViews(mesh_data, mesh_map_offset,
                            soln_data, soln_map_offset,
                            resid_data, resid_map_offset);
}

BOOST_AUTO_TEST_CASE(Evaluate) {
  Evaluator<double,double>* jacob = new MetricJacobian<double,double>();
  Evaluator<double,double>* laplace_pde = new Laplace<double,double>();
  int num_elems = 10;
  int dim = 2;

  // define a triangular-element topology and its cubature
  Teuchos::RCP<const CellTopologyData>
      cell(shards::getCellTopologyData<shards::Triangle<3> >(), false);
  RCP<shards::CellTopology> topology =
      Teuchos::rcp(new shards::CellTopology(cell.get()));
  int num_nodes_per_elem = topology->getNodeCount();
  Intrepid::DefaultCubatureFactory<double> cubFactory;
  int degree = 3;
  RCP<Intrepid::Cubature<double> >
      cub = cubFactory.create(*topology, degree);
  int num_cub_points = cub->getNumPoints();
  FieldContainer<double> cub_points(num_cub_points, dim);
  FieldContainer<double> cub_weights(num_cub_points);
  cub->getCubature(cub_points, cub_weights);

  // define the FE basis
  Intrepid::Basis_HGRAD_TRI_C1_FEM<double, FieldContainer<double> > basis;
  int num_ref_basis = basis.getCardinality();
  FieldContainer<double> vals(num_ref_basis, num_cub_points);
  FieldContainer<double> grads(num_ref_basis, num_cub_points, dim);
  basis.getValues(vals, cub_points, Intrepid::OPERATOR_VALUE);
  basis.getValues(grads, cub_points, Intrepid::OPERATOR_GRAD);

  // Determine memory requirements
  std::cout << "memory requirements...\n";
  jacob->SetDimensions(num_elems, num_nodes_per_elem, num_cub_points,
                       num_ref_basis, dim);
  laplace_pde->SetDimensions(num_elems, num_nodes_per_elem, num_cub_points,
                             num_ref_basis, dim);
  int mesh_offset = num_elems*num_nodes_per_elem*dim; // for nodes
  int soln_offset = num_elems*num_ref_basis; // for solution coeffs
  int resid_offset = 0;
  std::map<std::string,int> mesh_map_offset, soln_map_offset, resid_map_offset;
  mesh_map_offset["node_coords"] = 0;
  soln_map_offset["solution_coeff"] = 0;
  jacob->MemoryRequired(mesh_offset, mesh_map_offset,
                        soln_offset, soln_map_offset,
                        resid_offset, resid_map_offset);
  laplace_pde->MemoryRequired(mesh_offset, mesh_map_offset,
                          soln_offset, soln_map_offset,
                          resid_offset, resid_map_offset);

  // allocate memory and set data views
  Teuchos::ArrayRCP<double> mesh_data(mesh_offset, 1.0);
  Teuchos::ArrayRCP<double> soln_data(soln_offset, 1.0);
  Teuchos::ArrayRCP<typename davinci::Evaluator<double,double>::ResidT>
      resid_data(resid_offset, 1.0);
  jacob->SetDataViews(mesh_data, mesh_map_offset,
                      soln_data, soln_map_offset,
                      resid_data, resid_map_offset);
  laplace_pde->SetDataViews(mesh_data, mesh_map_offset,
                            soln_data, soln_map_offset,
                            resid_data, resid_map_offset);
  
  // define node coordinates; the elements do not need to be adjacent
  for (int i = 0; i < num_elems; i++) {
    // node 1
    mesh_data[i*num_nodes_per_elem*dim] = static_cast<double>(i); //< x
    mesh_data[i*num_nodes_per_elem*dim+1] = 0.0; //< y
    // node 2
    mesh_data[i*num_nodes_per_elem*dim+2] = static_cast<double>(i+1); //< x
    mesh_data[i*num_nodes_per_elem*dim+3] = 0.0; //< y
    // node 3
    mesh_data[i*num_nodes_per_elem*dim+4] = static_cast<double>(i); //< x
    mesh_data[i*num_nodes_per_elem*dim+5] = 1.0; //< y
  }
  // define the solution coefficients (u(x,y) = x)
  for (int i = 0; i < num_elems; i++) {
    // node 1
    soln_data[i*num_ref_basis] = static_cast<double>(i);
    // node 2
    soln_data[i*num_ref_basis+1] = static_cast<double>(i+1);
    // node 3
    soln_data[i*num_ref_basis+2] = static_cast<double>(i);
  }
  jacob->Evaluate(*topology, cub_points, cub_weights, vals, grads);
  laplace_pde->Evaluate(*topology, cub_points, cub_weights, vals, grads);
  // check that gradient is (1,0) at all cubature points
  for (int i = 0; i < num_elems; ++i)
    for (int j = 0; j < num_cub_points; ++j) {
      BOOST_CHECK_CLOSE(resid_data[resid_map_offset.at("solution_grad")
                                  + (i*num_cub_points + j)*dim], 1.0, 1e-13);
      BOOST_CHECK_SMALL(resid_data[resid_map_offset.at("solution_grad")
                                  + (i*num_cub_points + j)*dim + 1], 1e-13);
    }
  // check that residual is (-0.5, 0.5, 0.0) on each element; this follows
  // because the solution is linear, and, when the element residual vectors
  // are assembled, the residual becomes zero.
  for (int i = 0; i < num_elems; ++i) {
    BOOST_CHECK_CLOSE(resid_data[resid_map_offset.at("residual")
                                 + i*num_ref_basis], -0.5, 1e-13);
    BOOST_CHECK_CLOSE(resid_data[resid_map_offset.at("residual")
                                 + i*num_ref_basis + 1], 0.5, 1e-13);
    BOOST_CHECK_CLOSE(resid_data[resid_map_offset.at("residual")
                                 + i*num_ref_basis + 2], 0.0, 1e-13);
  }
}

BOOST_AUTO_TEST_CASE(BuildLinearSystemWorkSet) {
  RCP<WorkSetFactoryBase<SimpleMesh> > MyFactory = Teuchos::rcp(
      new WorkSetFactory<LaplaceFactory<SimpleMesh> >);
  RCP<const Basis<double, FieldContainer<double> > > basis = Teuchos::rcp(
      new Intrepid::Basis_HGRAD_TRI_C1_FEM<double, FieldContainer<double> >);
  Teuchos::Array<RCP<WorkSetBase<SimpleMesh> > > worksets;
  Teuchos::ParameterList options;
  options.set("cub degree",2);
  options.set("num local elems", 100);
  options.set("num elems per set", 10);
  MyFactory->BuildLinearSystemWorkSet(options, basis, worksets);
}

BOOST_AUTO_TEST_SUITE_END()
