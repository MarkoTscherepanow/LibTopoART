using System.Collections.Generic;

namespace LibTopoART
{

//**********************************************************************************************************************

#if DEBUG
	/// <summary>Struct describing a node's most important parameters.</summary>
	public readonly struct NodeDescriptor<TSpatialWeightType>
#else
	internal readonly struct NodeDescriptor<TSpatialWeightType>
#endif
	{
#if DEBUG
		/// <summary>A flag signalling that a node is permanent.</summary>
#endif
		public readonly bool _permanent;

#if DEBUG
		/// <summary>An array holding the weights of a node.</summary>
#endif
		public readonly  TSpatialWeightType[] _weights;

#if DEBUG
		/// <summary>This constructor initialises an instance of struct <c>NodeDescriptor</c>.</summary>
		/// <param name="nodeWeights">An array holding the weights of a node.</param>
		/// <param name="permanent">A flag signalling that a node is permanent.</param>
#endif
		public NodeDescriptor(TSpatialWeightType[] nodeWeights, bool permanent)
		{
			_permanent = permanent;
			_weights = new  TSpatialWeightType[nodeWeights.LongLength];
			nodeWeights.CopyTo(_weights, 0);
		}
	}

#if DEBUG
	/// <summary>Struct describing an edge's most important parameters.</summary>
	public readonly struct EdgeDescriptor
#else
	internal readonly struct EdgeDescriptor
#endif
	{
#if DEBUG
		/// <summary>A flag signalling that an edge is permanent.</summary>
#endif
		public readonly bool _permanent;

#if DEBUG
		/// <summary>This constructor initialises an instance of struct <c>EdgeDescriptor</c>.</summary>
		/// <param name="permanent">A flag signalling that an edge is permanent.</param>
#endif
		public EdgeDescriptor(bool permanent)
		{
			_permanent = permanent;
		}
	}

//**********************************************************************************************************************

#if DEBUG
	/// <summary>Class <c>Snapshot</c> stores a snapshot of the nodes and edges of a TopoART network.</summary>
	public class Snapshot<TSpatialWeightType>
#else
	internal class Snapshot<TSpatialWeightType>
#endif
	{
		private readonly Dictionary<(long, long), EdgeDescriptor> _edgeMap;
		private Dictionary<long, NodeDescriptor<TSpatialWeightType>> _nodeMap;

#if DEBUG
		/// <value>Property <c>EdgeMap</c> provides a dictionary containing all edges.</value>
#endif
		public Dictionary<(long, long), EdgeDescriptor> EdgeMap { get => _edgeMap; }

#if DEBUG
		/// <value>Property <c>NodeMap</c> provides a dictionary containing all nodes.</value>
#endif
		public Dictionary<long, NodeDescriptor<TSpatialWeightType>> NodeMap
		{
			get => _nodeMap;
			set => _nodeMap = value;
		}

#if DEBUG
		/// <summary>This constructor initialises an instance of class <c>Snapshot</c>.</summary>
		/// <param name="nodeState">Interface <c>NodeState</c> of the first node to be stored.</param>
		/// <param name="phi">The value of phi to be used.</param>
		/// <exception cref="InvalidNumberException">Thrown when the number of edges of an F2 node is greater than
		/// <c>int.MaxValue</c>.</exception>
#endif
		public Snapshot(IF2_node_state<TSpatialWeightType>? nodeState, long phi)
		{
			var tmpEdgeMap = new Dictionary<(long, long), EdgeDescriptor>();
			_edgeMap = new Dictionary<(long, long), EdgeDescriptor>();
			_nodeMap = new Dictionary<long, NodeDescriptor<TSpatialWeightType>>();

			for(var currentNodeState = nodeState; currentNodeState != null; currentNodeState = currentNodeState.StateNext)
			{
				_nodeMap.Add(currentNodeState.NodeID, new NodeDescriptor<TSpatialWeightType>(currentNodeState.Weights,
					!currentNodeState.IsNodeCandidate(phi)));
				var currentEdgeList = currentNodeState.Edges;

				foreach(var edge in currentEdgeList)
					tmpEdgeMap.Add(edge, new EdgeDescriptor(!currentNodeState.IsNodeCandidate(phi)));
			}

			// remove double edges and fuse permanent flags
			foreach(KeyValuePair<(long, long), EdgeDescriptor> pair in tmpEdgeMap)
			{
				if(pair.Key.Item1.CompareTo(pair.Key.Item2) < 0) {
					_edgeMap.Add(pair.Key, new EdgeDescriptor(pair.Value._permanent &&
						tmpEdgeMap[(pair.Key.Item2, pair.Key.Item1)]._permanent));
				}
			}
		}

#if DEBUG
		/// <summary>This method compares the current instance of class <c>Snapshot</c> to another one.</summary>
		/// <param name="other">The instance of class <c>Snapshot</c> the current instance is compared to.</param>
		/// <param name="weightCmpFunction">Function to be used for the comparison of single weights.</param>
		/// <exception cref="InvalidStateException">Thrown when the network is in an invalid state. This happens if a
		/// permanent node or an edge between two permanent nodes of <c>this</c> is not present in
		/// <paramref name="other"/>.</exception>
#endif
		public AdaptationState CompareTo(Snapshot<TSpatialWeightType> other,
										 CompareWeights<TSpatialWeightType> weightCmpFunction)
		{
			var state = AdaptationState.NO_ADAPTATION;

			var tmpNodeMap = _nodeMap == null ? new Dictionary<long, NodeDescriptor<TSpatialWeightType>>() :
				new Dictionary<long, NodeDescriptor<TSpatialWeightType>>(_nodeMap);

			var tmpOtherNodeMap = other?._nodeMap == null ? new Dictionary<long, NodeDescriptor<TSpatialWeightType>>() :
				new Dictionary<long, NodeDescriptor<TSpatialWeightType>>(other._nodeMap);

			foreach(var pair in tmpOtherNodeMap) {
				if(!tmpNodeMap.ContainsKey(pair.Key))
					state |= pair.Value._permanent ? AdaptationState.ADDED_PERMANENT_NODE : AdaptationState.ADDED_NODE_CANDIDATE;
				else {
					var nd1 = tmpNodeMap[pair.Key];
					var nd2 = pair.Value;

					for(long i = 0; i < nd1._weights.LongLength; ++i)
						if(weightCmpFunction(nd2._weights[i], nd1._weights[i]))
							state |= pair.Value._permanent ? AdaptationState.ADAPTED_PERMANENT_WEIGHT : AdaptationState.ADAPTED_NONPERMANENT_WEIGHT;

					tmpNodeMap.Remove(pair.Key);
				}
			}

			foreach(KeyValuePair<long, NodeDescriptor<TSpatialWeightType>> pair in tmpNodeMap) {
				if(!pair.Value._permanent)
					state |= AdaptationState.REMOVED_NODE_CANDIDATE;
				else
					throw new InvalidStateException(Common.InvalidStateException_InvalidNetworkState);
			}

			var tmpEdgeMap = _edgeMap == null ? new Dictionary<(long, long), EdgeDescriptor>() :
				new Dictionary<(long, long), EdgeDescriptor>(_edgeMap);

			var tmpOtherEdgeMap = other?._edgeMap == null ? new Dictionary<(long, long), EdgeDescriptor>() :
				new Dictionary<(long, long), EdgeDescriptor>(other._edgeMap);

			foreach(KeyValuePair<(long, long), EdgeDescriptor> pair in tmpOtherEdgeMap) {
				if(!tmpEdgeMap.ContainsKey(pair.Key))
					state |= pair.Value._permanent ? AdaptationState.ADDED_PERMANENT_EDGE : AdaptationState.ADDED_EDGE_CANDIDATE;
				else
					tmpEdgeMap.Remove(pair.Key);
			}

			foreach(KeyValuePair<(long, long), EdgeDescriptor> pair in tmpEdgeMap) {
				if(!pair.Value._permanent)
					state |= AdaptationState.REMOVED_EDGE_CANDIDATE;
				else
					throw new InvalidStateException(Common.InvalidStateException_InvalidNetworkState);
			}

			return state;
		}
	}

//**********************************************************************************************************************

}