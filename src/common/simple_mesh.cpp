/**
 * \file simple_mesh.cpp
 * \brief class definitions for SimpleMesh
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#include <string>
#include <boost/assert.hpp>
#include "Teuchos_Tuple.hpp"
#include "simple_mesh.hpp"

namespace davinci {
//==============================================================================
SimpleMesh::SimpleMesh(ostream & out, const RCP<const Comm<int> >& comm) :
    node_coord_(),
    node_type_(),
    elem_to_node_(),
    comm_(comm),
    out_(&out, false) {
}
//==============================================================================
void SimpleMesh::Initialize(ParameterList& p) {
#ifdef DAVINCI_VERBOSE
  *out_ << "SimpleMesh: initializing the mesh\n";
#endif
  
  if (p.get<std::string>("Mesh Type") == "Rectangular") {
    BuildRectangularMesh(p.get("Lx", 1.0), p.get("Ly", 1.0),
                         p.get("Nx", 2), p.get("Ny", 2));
  } else {
    *out_ << "Error in SimpleMesh::Initialize: "
          << "invalid Mesh Type in parameterlist.\n";
  }
}
//==============================================================================
void SimpleMesh::BuildRectangularMesh(const double& Lx, const double& Ly,
                                      const int & Nx, const int & Ny) {
  BOOST_ASSERT_MSG(Lx > 0.0 && Ly > 0.0, "Lx and Ly must be positive");
  BOOST_ASSERT_MSG(Nx > 0 && Ny > 0, "Nx and Ny must be positive");
  BOOST_ASSERT_MSG(comm_->getSize() == 1, "Not generalized for parallel yet");
  dim_ = 2;
  num_elems_ = 2*Nx*Ny;
  num_nodes_ = (Nx+1)*(Ny+1);
#ifdef DAVINCI_VERBOSE
  *out_ << "Generating triangular mesh on rectangular domain... \n\n";
  *out_ << "   Nx" << "   Ny\n";
  *out_ << std::setw(5) << Nx << std::setw(5) << Ny <<"\n\n";
  *out_ << " Number of Elements: " << num_elems_ << " \n";
  *out_ << " Number of Nodes:    " << num_nodes_ << " \n\n";
#endif

  // Define node coordinates, set indices, and determine boundary nodes
  double x0 = 0.0;
  double y0 = 0.0;
  double hx = 1.0/static_cast<double>(Nx);
  double hy = 1.0/static_cast<double>(Ny);
  node_coord_.resize(num_nodes_, dim_);
  node_type_.resize(num_nodes_);
  index_.resize(num_nodes_);
  int inode = 0;
  for (int j=0; j<Ny+1; j++) {
    for (int i=0; i<Nx+1; i++) {
      index_(inode) = inode;
      node_coord_(inode,0) = x0 + (double)i*hx;
      node_coord_(inode,1) = y0 + (double)j*hy;
      if (j==0 || i==0 || j==Ny || i==Nx) {
        node_type_(inode) = 1;
      } else {
        node_type_(inode) = 0;
      }
      inode++;
    }
  }

  // Element to Node map
  int num_nodes_per_elem = 3;
  elem_to_node_.resize(num_elems_, num_nodes_per_elem);
  int ielem = 0;
  for (int j=0; j<Ny; j++) {
    for (int i=0; i<Nx; i++) {
      // first triangle
      elem_to_node_(ielem,0) = (Nx + 1)*j + i;
      elem_to_node_(ielem,1) = (Nx + 1)*j + i + 1;
      elem_to_node_(ielem,2) = (Nx + 1)*(j + 1) + i;
      ielem++;
      // second triangle
      elem_to_node_(ielem,0) = (Nx + 1)*j + i + 1;
      elem_to_node_(ielem,1) = (Nx + 1)*(j + 1) + i + 1;
      elem_to_node_(ielem,2) = (Nx + 1)*(j + 1) + i;
      ielem++;
    }
  }
}
//==============================================================================
void SimpleMesh::BuildTpetraMap(RCP<const Map<LocIdxT,GlbIdxT> >& map) const {
  BOOST_ASSERT_MSG(index_.size() > 0, "Mesh must be initialized");
  BOOST_ASSERT_MSG(comm_->getSize() == 1, "Not generalized for parallel yet");
#ifdef DAVINCI_VERBOSE
  *out_ << "SimpleMesh: creating Tpetra Map\n";
#endif
  map = Tpetra::createContigMap<LocIdxT,GlbIdxT>(-1, num_nodes_, comm_);
#ifdef DAVINCI_VERBOSE
  *out_ << "SimpleMesh: total number of nodes (global) = "
        << map->getGlobalNumElements() << "\n";
#endif
}
//==============================================================================
void SimpleMesh::BuildMatrixGraph(const RCP<const Map<LocIdxT,GlbIdxT> >& map,
                                  RCP<CrsGraph<LocIdxT,GlbIdxT> >& jac_graph
                                  ) const {
  BOOST_ASSERT_MSG(index_.size() > 0, "Mesh must be initialized");
  BOOST_ASSERT_MSG(comm_->getSize() == 1, "Not generalized for parallel yet");
#ifdef DAVINCI_VERBOSE
  *out_ << "SimpleMesh: creating Graph for Jacobian\n";
#endif
  jac_graph = Tpetra::createCrsGraph<LocIdxT,GlbIdxT>(map);
  int num_nodes_per_elem = 3;
  using Teuchos::Tuple;
  using Teuchos::tuple;
  for (int ielem = 0; ielem < num_elems_; ielem++) {
  // insert rows corresponding to nodes making up this element
  Tuple<int, 3> indices = tuple(index_(elem_to_node_(ielem, 0)),
      elem_to_node_(ielem,1), elem_to_node_(ielem,2));
  for (int i = 0; i < num_nodes_per_elem; i++)
    jac_graph->insertGlobalIndices(elem_to_node_(ielem, i), indices);
}
  jac_graph->fillComplete();
#ifdef DAVINCI_VERBOSE
  jac_graph->print(*out_);
#endif
}
//==============================================================================
void SimpleMesh::CopyElemNodeCoords(ArrayRCP<double>& node_coords,
                                    const int& set_idx,
                                    const int& num_elems_per_set,
                                    const int& num_sets) const {
  BOOST_ASSERT_MSG(set_idx >= 0 && set_idx < num_sets,
                   "set_idx number must be positive and less than num_sets_");
  // copy the physical cell coordinates
  int set_num_elems = num_elems_per_set;
  if (set_idx == num_sets-1)
    set_num_elems = num_elems_ - (set_idx*num_elems_per_set);
  int num_nodes_per_elem = 3;
  for (int ielem = 0; ielem < set_num_elems; ielem++) {
    int k = set_idx*num_elems_per_set + ielem;
    for (int i = 0; i < num_nodes_per_elem; i++)
      for (int j = 0; j < dim_; j++)
        node_coords[(ielem*num_nodes_per_elem+i)*dim_ + j]
            = node_coord_(elem_to_node_(k, i), j);
  }
}
//==============================================================================
void SimpleMesh::CopyElemDOFIndices(ArrayRCP<LocIdxT>& dof_index,
                                    const int& set_idx,
                                    const int& num_elems_per_set,
                                    const int& num_sets,
                                    const int& num_pdes) const {
  BOOST_ASSERT_MSG(set_idx >= 0 && set_idx < num_sets,
                   "set_idx number must be positive and less than num_sets_");
  // set the DOF indices
  int set_num_elems = num_elems_per_set;
  if (set_idx == num_sets-1)
    set_num_elems = num_elems_ - (set_idx*num_elems_per_set);
  int num_nodes_per_elem = 3;
  for (int ielem = 0; ielem < set_num_elems; ielem++) {
    int k = set_idx*num_elems_per_set + ielem;
    for (int i = 0; i < num_nodes_per_elem; i++) {
      LocIdxT node_idx = index_(elem_to_node_(k, i));
      for (int j = 0; j < num_pdes; j++)
        dof_index[(ielem*num_nodes_per_elem+i)*num_pdes + j]
            = num_pdes*node_idx + j;
    }
  }
}
//==============================================================================
} // namespace davinci
