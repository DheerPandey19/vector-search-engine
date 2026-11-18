/// Edges of a Navigable Small World graph.
///
/// `neighbors[i]` is the list of point indexes that point `i` links to.
/// The vectors stay in the caller's `points` slice. This struct stores edges only.

pub struct NswGraph {
    neighbors : Vec<Vec<usize>>,
}


// Inherent implementation block
impl NswGraph {
    // n is how many points exist in the graqph
    // every point starts with an empty "neighbor list"

    pub fn new(n: usize) ->Self {
        Self {
            neighbors : vec![vec![]; n],
        }
    }

    pub fn len(&self) -> usize {
        self.neighbors.len()
    }

    /// Borrow the neighbor indexes of one point.
    pub fn neighbors(&self, index: usize) -> &[usize] {
        &self.neighbors[index]
    }
    
    pub fn add_bidirectional(&mut self, a : usize, b: usize){
        if a==b{
            return;
        }
        self.add_one_way(a, b);
        self.add_one_way(b, a);
    }
    fn add_one_way(&mut self,from: usize, to: usize){
        let list = &mut self.neighbors[from];
        if !list.contains(&to){
            list.push(to);
        }
    }

}