using System;
using System.IO;
using System.Diagnostics;
using System.Globalization;

namespace LibTopoART
{

//**********************************************************************************************************************

	internal class TA_F2_node :
		F2_edges,
		IF2_node_threading<decimal, decimal, long, bool>,
		IF2_node_state<decimal>
	{
		// used only for code simplification
		private const long UNDEFINED = LibTopoART_info.UNDEFINED;

		private protected readonly long _inputLen;
		private protected long _representedInputs;
		private protected decimal[] _weights;
		private protected decimal _weightsSum = UNDEFINED;
		private protected decimal _sizeCache = UNDEFINED;

		internal TA_F2_node? _next;

//----------------------------------------------------------------------------------------------------------------------

		public decimal Activation { get; set; }

		public decimal[] CentreOfGravity
		{
			get {
				var d = _inputLen >> 1;
				var cog = new decimal[d];

				for(long i = 0; i < d; ++i)
					cog[i] = 0.5m * (_weights[i] + 1.0m - _weights[d + i]);

				return cog;
			}
		}

		public virtual long ClassID { get => LibTopoART_info.UNDEFINED; }
		public long ClusterID { get; set; }
		public decimal MatchValue { get; set; }
		public IF2_node<decimal, decimal, long, bool>? Next { get => _next; }
		public long NodeID { get; }
		public IF2_node_state<decimal>? StateNext { get => _next; }
		public IF2_node_threading<decimal, decimal, long, bool>? ThreadNext { get; set; }
		public IF2_node_threading<decimal, decimal, long, bool>? ThreadPrev { get; set; }
		public long ThreadID { get; set; }
		public decimal[] Weights { get => _weights; }

//----------------------------------------------------------------------------------------------------------------------

		public TA_F2_node(long nodeID, long inputLen, decimal[] initialWeights)
		{
			NodeID = nodeID;
			_inputLen = inputLen;
			_representedInputs = 1;

			InitEdges(nodeID);

			_weights = new decimal[_inputLen];
			for(long i = 0; i < _inputLen; ++i)
				_weights[i] = initialWeights[i];

			Activation = UNDEFINED;
			MatchValue = UNDEFINED;
			ClusterID = UNDEFINED;

			_next = null;

			ThreadPrev = null;
			ThreadNext = null;
			ThreadID = 0;
		}

		public TA_F2_node(BinaryReader reader, in FileFormatVersions fileFormatVersions)
		{
			NodeID = reader.ReadInt64();
			_inputLen = reader.ReadInt64();
			_representedInputs = reader.ReadInt64();
			var edgeNum = reader.ReadInt64();

			InitEdges(NodeID, edgeNum, reader, fileFormatVersions.TopoARTFileFormatVersion);

			_weights = new decimal[_inputLen];
			for(long i = 0; i < _inputLen; ++i)
				_weights[i] = reader.ReadDecimal();

			Activation = reader.ReadDecimal();
			MatchValue = reader.ReadDecimal();
			ClusterID = reader.ReadInt64();

			LoadAdditionalData(reader, fileFormatVersions);

			_next = null;

			ThreadPrev = null;
			ThreadNext = null;
			ThreadID = 0;
		}

		private protected virtual void LoadAdditionalData(BinaryReader reader, in FileFormatVersions fileFormatVersions) {}

//----------------------------------------------------------------------------------------------------------------------

		public void PrintWeights()
		{
			for(long i = 0; i < _inputLen; ++i) {
				if(i != 0)
					Console.Write(" ");
				Console.Write(_weights[i]);
			}
			Console.Write("\n");
		}

		public void AdaptWeights(decimal[] x_F1, decimal beta)
		{
			AdaptWeightsInternal(x_F1, beta);
		}

		private protected virtual void AdaptWeightsInternal(decimal[] x_F1, decimal beta)
		{
			++_representedInputs;

			if(beta == 1.0m) {
				for(long i = 0; i < _inputLen; ++i)
					_weights[i] = Math.Min(x_F1[i], _weights[i]);

				_weightsSum = UNDEFINED;
				_sizeCache = UNDEFINED;
			} else if(beta != 0.0m) {
				var betaNeg = 1.0m - beta;

				// reduce each result to its minimal scale without changing its value
				for(long i = 0; i < _inputLen; ++i)
					_weights[i] = (beta * (Math.Min(x_F1[i], _weights[i])) + betaNeg * _weights[i]) / 1.0000000000000000000000000000m;

				_weightsSum = UNDEFINED;
				_sizeCache = UNDEFINED;
			}
		}

		public void ComputeAlternativeChoiceFunction(decimal[] x_F1, bool[]? mask)
		{
			ComputeAlternativeChoiceFunctionInternal(x_F1, mask);
		}

		private protected virtual decimal ComputeAlternativeChoiceFunctionInternal(decimal[] x_F1, bool[]? mask)
		{
			Debug.Assert((x_F1.LongLength % 2) == 0);
			Debug.Assert((mask == null) || (x_F1.LongLength == mask.LongLength * 2));

			var d = _inputLen >> 1;
			var diffSum = 0.0m;

			// (x < w) ? w - x : 0 (element-wise) equals abs(min(x, w) - w)
			if(mask == null) {
				for(long i = 0; i < _inputLen; ++i) {
					if(x_F1[i] < _weights[i])
						diffSum += _weights[i] - x_F1[i];
					++i;
					if(x_F1[i] < _weights[i])
						diffSum += _weights[i] - x_F1[i];
				}
				Activation = 1.0m - diffSum / d;
			} else {
				long diffNum = 0;

				for(long i = 0 ; i < d; ++i) {
					if(mask[i] == false) {
						if(x_F1[i] < _weights[i])
							diffSum += _weights[i] - x_F1[i];
						++diffNum;
					}
				}
				for(long i = d; i < _inputLen; ++i) {
					if(mask[i - d] == false) {
						if(x_F1[i] < _weights[i])
							diffSum += _weights[i] - x_F1[i];
						++diffNum;
					}
				}

				Activation = (diffNum == 0) ? LibTopoART_info.UNDEFINED : (1.0m - diffSum / (0.5m * diffNum));
			}
			MatchValue = LibTopoART_info.UNDEFINED;

			return Activation;
		}

		public void ComputeChoiceAndMatchFunction(decimal[] x_F1, long[]? t_F1, bool[]? mask, decimal alpha)
		{
			Debug.Assert(t_F1 == null);
			ComputeSpatialChoiceAndMatchFunction(x_F1, mask, alpha);
		}

		private protected virtual void ComputeSpatialChoiceAndMatchFunction(decimal[] x_F1, bool[]? mask, decimal alpha)
		{
			Debug.Assert((x_F1.LongLength % 2) == 0);
			Debug.Assert((mask == null) || (x_F1.LongLength == mask.LongLength * 2));
			Debug.Assert(x_F1.LongLength == _weights.LongLength);

			decimal minSum = 0.0m;
			if(_weightsSum == UNDEFINED) {
				_weightsSum = 0.0m;
				for(long i = 0; i < _inputLen; ++i) {
					_weightsSum	+=	_weights[i];
					minSum			+=	Math.Min(x_F1[i], _weights[i]);
					++i;
					_weightsSum	+=	_weights[i];
					minSum			+=	Math.Min(x_F1[i], _weights[i]);
				}
			} else {
				for(long i = 0; i < _inputLen; ++i) {
					minSum += Math.Min(x_F1[i], _weights[i]);
					++i;
					minSum += Math.Min(x_F1[i], _weights[i]);
				}
			}

			Activation = minSum / (alpha + _weightsSum);

			if(mask == null) {
				MatchValue = minSum / (_inputLen >> 1);
			} else {
				var d = _inputLen >> 1;

				var partial_x_F1_sum_false = 0.0m;
				var partialMinSumFalse = 0.0m;
				var partial_x_F1_sum_true = 0.0m;
				var partialMinSumTrue = 0.0m;

				decimal partialMatchValueFalse;
				decimal partialMatchValueTrue;
				for(long i = 0; i < d; ++i) {
					if(mask[i] == false) {
						partial_x_F1_sum_false += x_F1[i];
						partialMinSumFalse += Math.Min(x_F1[i], _weights[i]);
					} else {
						partial_x_F1_sum_true += x_F1[i];
						partialMinSumTrue += Math.Min(x_F1[i], _weights[i]);
					}
				}
				for(var i = d; i < _inputLen; ++i) {
					if(mask[i - d] == false) {
						partial_x_F1_sum_false += x_F1[i];
						partialMinSumFalse += Math.Min(x_F1[i], _weights[i]);
					} else {
						partial_x_F1_sum_true += x_F1[i];
						partialMinSumTrue += Math.Min(x_F1[i], _weights[i]);
					}
				}

				if(partial_x_F1_sum_false > 0)
					partialMatchValueFalse = partialMinSumFalse / partial_x_F1_sum_false;
				else
					partialMatchValueFalse = 1.0m;

				if(partial_x_F1_sum_true > 0)
					partialMatchValueTrue = partialMinSumTrue / partial_x_F1_sum_true;
				else
					partialMatchValueTrue = 1.0m;

				MatchValue = Math.Min(partialMatchValueFalse, partialMatchValueTrue);
			}
		}

		public virtual decimal GetCombinedChoiceAndMatchValue(MatchFunction<TA_F2_node, decimal>? matchFunction, decimal rho)
		{
			return ((matchFunction != null) && matchFunction(this, rho) ? Activation : -1.0m);
		}

		public decimal[] GetCopyOfSpatialWeights()
		{
			decimal[] weightsCopy = new decimal[_weights.LongLength];
			_weights.CopyTo(weightsCopy, 0);
			return weightsCopy;
		}

		public virtual decimal[]? GetCopyOfTemporalWeights()
		{
			return null;
		}

		public bool IsNodeCandidate(long phi)
		{
			return (_representedInputs < phi);
		}

//----------------------------------------------------------------------------------------------------------------------

		public void SaveText(TextWriter writer)
		{
			writer.WriteLine("------------ F2 node ------------");

			writer.WriteLine("node ID: " + NodeID);
			writer.WriteLine("cluster ID: " + ClusterID);
			writer.WriteLine("input length: " + _inputLen);
			writer.WriteLine("represented inputs: " + _representedInputs);
			writer.WriteLine("activation: " + Activation.ToString(CultureInfo.InvariantCulture));
			writer.WriteLine("match value: " + MatchValue.ToString(CultureInfo.InvariantCulture));

			SaveAdditionalText(writer);

			writer.WriteLine("............. edges .............");

			SaveEdgesText(writer, Common.TopoART_file_format_version);

			writer.WriteLine("............ weights ............");
			for(long i = 0; i < _inputLen; ++i) {
				if(i != 0) writer.Write(" ");
				writer.Write(_weights[i].ToString(CultureInfo.InvariantCulture));
			}
			writer.Write("\n");
		}

		private protected virtual void SaveAdditionalText(TextWriter writer) {}

		public void Save(BinaryWriter writer)
		{
			writer.Write(NodeID);
			writer.Write(_inputLen);
			writer.Write(_representedInputs);
			SaveEdges(writer, Common.TopoART_file_format_version);

			for(long i = 0; i < _inputLen; ++i)
				writer.Write(_weights[i]);

			writer.Write(Activation);
			writer.Write(MatchValue);
			writer.Write(ClusterID);

			SaveAdditionalData(writer);
		}

		private protected virtual void SaveAdditionalData(BinaryWriter writer) {}
	}

//**********************************************************************************************************************

	internal class HTA_F2_node : TA_F2_node
	{
		protected decimal R { get; }

//----------------------------------------------------------------------------------------------------------------------

		public HTA_F2_node(long nodeID, long inputLen, decimal[] initialWeights, decimal R)
			: base(nodeID, inputLen, initialWeights)
		{
			this.R = R;
		}

		public HTA_F2_node(BinaryReader reader, in FileFormatVersions fileFormatVersions, decimal R)
			: base(reader, fileFormatVersions)
		{
			this.R = R;
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected override void AdaptWeightsInternal(decimal[] x_F1, decimal beta)
		{
			long i;
			double squaredSum;

			// compute distance
			for(i = 0, squaredSum = 0; i < _inputLen - 1; ++i) {
				var diff = (double)(x_F1[i] - _weights[i]);
				squaredSum += diff * diff;
			}

			var dist = (decimal)Math.Sqrt(squaredSum);

			// adapt centre
			if(dist != 0.0m) {
				var factor = beta / 2.0m * (1.0m - Math.Min(_weights[_inputLen - 1], dist) / dist);

				// reduce each result to its minimal scale without changing its value
				for(i = 0; i < _inputLen - 1; ++i)
					_weights[i] = (_weights[i] + factor * (x_F1[i] - _weights[i])) / 1.0000000000000000000000000000m;
			}

			// adapt radius
			_weights[_inputLen - 1] = (_weights[_inputLen - 1] + beta / 2.0m * (Math.Max(_weights[_inputLen - 1], dist) - _weights[_inputLen - 1]))
				/ 1.0000000000000000000000000000m;

			++_representedInputs;
		}

		private protected override decimal ComputeAlternativeChoiceFunctionInternal(decimal[] x_F1, bool[]? mask)
		{
			long i;
			double squaredSum;
			decimal dist;

			// compute distance
			if(mask == null) {
				for(i = 0, squaredSum = 0; i < _inputLen - 1; ++i) {
					var diff = (double)(x_F1[i] - _weights[i]);
					squaredSum += diff * diff;
				}
				dist = (decimal)Math.Sqrt(squaredSum);
				Activation = Math.Max(1.0m - Math.Max(dist - _weights[_inputLen - 1], 0.0m) / (2.0m * R), 0.0m);
			} else {
				long diffNum = 0;

				for(i = 0, squaredSum = 0; i < _inputLen - 1; ++i) {
					if(mask[i] == false) {
						var diff = (double)(x_F1[i] - _weights[i]);
						squaredSum += diff * diff;
						++diffNum;
					}
				}
				dist = (decimal)Math.Sqrt(squaredSum);

				var partialR = (decimal)Math.Sqrt((double)(((R * R) / (_inputLen - 1)) * diffNum));
				Activation = (diffNum == 0) ? -1.0m : Math.Max((1.0m - Math.Max(dist - _weights[_inputLen - 1], 0.0m) / (2.0m * partialR)), 0.0m);
			}
			MatchValue = -1.0m;

			return Activation;
		}

		private protected override void ComputeSpatialChoiceAndMatchFunction(decimal[] x_F1, bool[]? mask, decimal alpha)
		{
			long i;
			double squaredSum;

			// compute distance
			if(mask == null) {
				for(i = 0, squaredSum = 0; i < _inputLen - 1; ++i) {
					var diff = (double)(x_F1[i] - _weights[i]);
					squaredSum += diff * diff;
				}
			} else {
				for(i = 0, squaredSum = 0; i < _inputLen - 1; ++i) {
					if(mask[i] == false) {
						var diff = (double)(x_F1[i] - _weights[i]);
						squaredSum += diff * diff;
					}
				}
			}
			var dist = (decimal)Math.Sqrt(squaredSum);
			var max = Math.Max(_weights[_inputLen - 1], dist);

			Activation	= (R - max) / (R - _weights[_inputLen - 1] + alpha);
			MatchValue	= 1.0m - (max / R);
		}
	}

//**********************************************************************************************************************

	internal sealed class TAC_F2_node : TA_F2_node
	{
		private long _classID;

		public override long ClassID { get => _classID; }

		public decimal Size
		{
			get {

				if(_sizeCache == LibTopoART_info.UNDEFINED) {
					long i;
					decimal size;
					var d = _inputLen >> 1;

					for(i = 0, size = 0m; i < d; ++i) {
						size += Math.Abs((1m - _weights[d + i]) - _weights[i]);
					}

					_sizeCache = size;
				}

				return _sizeCache;
			}
		}

//----------------------------------------------------------------------------------------------------------------------

		public TAC_F2_node(long nodeID, long inputLen, decimal[] initialWeights, long classID)
			: base(nodeID, inputLen, initialWeights)
		{
			_classID = classID;
		}

		public TAC_F2_node(BinaryReader reader, in FileFormatVersions fileFormatVersions)
			: base(reader, fileFormatVersions) {}

		private protected override void LoadAdditionalData(BinaryReader reader, in FileFormatVersions fileFormatVersions)
		{
			_classID = reader.ReadInt64();
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected override void SaveAdditionalText(TextWriter writer)
		{
			writer.WriteLine("class ID: " + ClassID);
		}

		private protected override void SaveAdditionalData(BinaryWriter writer)
		{
			writer.Write(ClassID);
		}
	}

//**********************************************************************************************************************

	internal sealed class HTAC_F2_node : HTA_F2_node
	{
		private long _classID;

		public override long ClassID { get => _classID; }
		public decimal Size { get => _weights[_inputLen - 1]; }

//----------------------------------------------------------------------------------------------------------------------

		public HTAC_F2_node(long nodeID, long inputLen, decimal[] initialWeights, decimal R, long classID)
			: base(nodeID, inputLen, initialWeights, R)
		{
			_classID = classID;
		}

		public HTAC_F2_node(BinaryReader reader, in FileFormatVersions fileFormatVersions, decimal R)
			: base(reader, fileFormatVersions, R) {}

		private protected override void LoadAdditionalData(BinaryReader reader, in FileFormatVersions fileFormatVersions)
		{
			_classID = reader.ReadInt64();
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected override void SaveAdditionalText(TextWriter writer)
		{
			writer.WriteLine("class ID: " + ClassID);
		}

		private protected override void SaveAdditionalData(BinaryWriter writer)
		{
			writer.Write(ClassID);
		}
	}

//**********************************************************************************************************************

}