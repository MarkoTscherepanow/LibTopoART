using System;
using System.IO;
using System.Diagnostics;
using System.Globalization;
using System.Numerics;

namespace LibTopoART
{

//**********************************************************************************************************************

	internal class FTA_F2_node :
		F2_edges,
		IF2_node_threading<int, Vector<int>, long, Vector<int>>,
		IF2_node_state<Vector<int>>
	{
		// used only for code simplification
		private const long UNDEFINED = LibTopoART_info.UNDEFINED;

		protected readonly long _inputLen;
		private long _representedInputs;
		protected Vector<int>[] _weights;
		protected int _matchValue;
		private long _weightsSum = UNDEFINED;

		internal FTA_F2_node? _next;

//----------------------------------------------------------------------------------------------------------------------

		public int Activation { get; private protected set; }

		public byte[] ByteAccessCentreOfGravity
		{
			get {
				var d = _inputLen >> 1;
				var dSimd = _weights.LongLength >> 1;
				var cog = new byte[d];

				Debug.Assert(dSimd == Common.SimdLength<int>(d));

				for(long i = 0; i < dSimd; ++i) {
					Vector<int> tmpCog = _weights[i] - _weights[dSimd + i] + Common.ScalingVectorInt;

					for(int j = 0; (j < Vector<int>.Count) && ((i * Vector<int>.Count + j) < d); ++j)
						cog[i * Vector<int>.Count + j] = (byte)((tmpCog[j] >> 1) / (int)Common.ScalingFactorByte);
				}

				return cog;
			}
		}

		public decimal[] DecimalAccessCentreOfGravity
		{
			get {
				var d = _inputLen >> 1;
				var dSimd = _weights.LongLength >> 1;
				var cog = new decimal[d];

				Debug.Assert(dSimd == Common.SimdLength<int>(d));

				for(long i = 0; i < dSimd; ++i) {
					Vector<int> tmpCog = _weights[i] - _weights[dSimd + i] + Common.ScalingVectorInt;

					for(int j = 0; (j < Vector<int>.Count) && ((i * Vector<int>.Count + j) < d); ++j)
						cog[i * Vector<int>.Count + j] = (tmpCog[j] >> 1) / (decimal)Common.ScalingFactor;
				}

				return cog;
			}
		}

		public virtual long ClassID { get => LibTopoART_info.UNDEFINED; }
		public long ClusterID { get; set; }

		public virtual int MatchValue
		{
			get => _matchValue;
			set => _matchValue = value;
		}

		public IF2_node<int, Vector<int>, long, Vector<int>>? Next { get => _next; }
		public long NodeID { get; }
		public IF2_node_state<Vector<int>>? StateNext { get => _next; }
		public IF2_node_threading<int, Vector<int>, long, Vector<int>>? ThreadNext { get; set; }
		public IF2_node_threading<int, Vector<int>, long, Vector<int>>? ThreadPrev { get; set; }
		public long ThreadID { get; set; }
		public Vector<int>[] Weights { get => _weights; }

//----------------------------------------------------------------------------------------------------------------------

		public FTA_F2_node(long nodeID, long inputLen, Vector<int>[] initialWeights)
		{
			NodeID = nodeID;
			_inputLen = inputLen;
			_representedInputs = 1;

			InitEdges(nodeID);

			_weights = new Vector<int>[initialWeights.LongLength];
			for(long i = 0; i < initialWeights.LongLength; ++i)
				_weights[i] = initialWeights[i];

			Activation = (int)-Common.ScalingFactor;
			_matchValue = (int)-Common.ScalingFactor;
			ClusterID = UNDEFINED;

			_next = null;

			ThreadPrev = null;
			ThreadNext = null;
			ThreadID = 0;
		}

		public FTA_F2_node(BinaryReader reader, in (FileFormatVersions fileFormatVersions, bool compatibilityMode) fileFormatInfo)
		{
			NodeID = reader.ReadInt64();
			_inputLen = reader.ReadInt64();
			_representedInputs = reader.ReadInt64();
			var edgeNum = reader.ReadInt64();

			InitEdges(NodeID, edgeNum, reader, fileFormatInfo.fileFormatVersions.TopoARTFileFormatVersion);

			var tmpWeights = new int[_inputLen];
			_weights = new Vector<int>[Common.SimdLengthEncoded<int>(_inputLen)];

			if(fileFormatInfo.compatibilityMode) {
				for(long i = 0; i < _inputLen; ++i)
					tmpWeights[i] = (int)(reader.ReadDecimal() * Common.ScalingFactor);

				Activation = (int)(reader.ReadDecimal() * Common.ScalingFactor);
				_matchValue = (int)(reader.ReadDecimal() * Common.ScalingFactor);
			} else {
				for(long i = 0; i < _inputLen; ++i)
					tmpWeights[i] = reader.ReadInt32();

				Activation = reader.ReadInt32();
				_matchValue = reader.ReadInt32();
			}

			_weights = Common.CreateEncodedVectorArray(tmpWeights)!;

			ClusterID = reader.ReadInt64();

			LoadAdditionalData(reader, fileFormatInfo);

			_next = null;

			ThreadPrev = null;
			ThreadNext = null;
			ThreadID = 0;
		}

		protected virtual void LoadAdditionalData(BinaryReader reader, in (FileFormatVersions, bool) fileFormatInfo) {}

//----------------------------------------------------------------------------------------------------------------------

		public void PrintWeights()
		{
			for(long i = 0; i < _inputLen; ++i) {
				if(i != 0)
					Console.Write(" ");
				var (i1, i2) = Common.SimdIndexes<int>(i, _inputLen >> 1);
				Console.Write(_weights[i1][i2] / (decimal)Common.ScalingFactor);
			}
			Console.Write("\n");
		}

		public virtual void AdaptWeights(Vector<int>[] x_F1, int beta)
		{
			var betaNeg = Common.ScalingFactor - beta;

			Debug.Assert(x_F1.LongLength == Common.SimdLengthEncoded<int>(_inputLen));

#if DEBUG
			var d = _inputLen >> 1;
			var diff = Common.SimdLength<int>(_inputLen >> 1) * Vector<int>.Count - d;
			var (i1, i2) = Common.SimdIndexes<int>(d - 1, d);
			var (i3, i4) = Common.SimdIndexes<int>(d + d - 1, d);
			for(long off = 0; off < diff; ++off) {
				Debug.Assert(x_F1[i1][(int)(i2 + 1 + off)] == 0, "invalid original input: " + x_F1[i1][(int)(i2 + 1 + off)].ToString());
				Debug.Assert(x_F1[i3][(int)(i4 + 1 + off)] == 0, "invalid encoded input: " + x_F1[i3][(int)(i4 + 1 + off)].ToString());
				Debug.Assert(i2 == i4);
			}
#endif

			++_representedInputs;
			if(beta == (int)Common.ScalingFactor) {
				for(long i = 0; i < x_F1.LongLength; ++i)
					_weights[i] = Vector.Min(x_F1[i], _weights[i]);
			} else if(beta != 0) {
				for(long i = 0; i < x_F1.LongLength; ++i) {
					Vector.Widen(Vector.Min(x_F1[i], _weights[i]), out Vector<long> tmp1, out Vector<long> tmp2);
					tmp1 = (tmp1 * beta) / Common.ScalingVectorLong;
					tmp2 = (tmp2 * beta) / Common.ScalingVectorLong;
					Vector<int> tmp = Vector.Narrow(tmp1, tmp2);
					Vector.Widen(_weights[i], out tmp1, out tmp2);
					tmp1 = (tmp1 * betaNeg) / Common.ScalingVectorLong;
					tmp2 = (tmp2 * betaNeg) / Common.ScalingVectorLong;
					_weights[i] = tmp + Vector.Narrow(tmp1, tmp2);
				}
			}

			_weightsSum = UNDEFINED;

#if DEBUG
			(i1, i2) = Common.SimdIndexes<int>(d - 1, d);
			(i3, i4) = Common.SimdIndexes<int>(d + d - 1, d);
			for(long off = 0; off < diff; ++off) {
				Debug.Assert(_weights[i1][(int)(i2 + 1 + off)] == 0, "invalid original weight: " + _weights[i1][(int)(i2 + 1 + off)].ToString());
				Debug.Assert(_weights[i3][(int)(i4 + 1 + off)] == 0, "invalid encoded weight: " + _weights[i3][(int)(i4 + 1 + off)].ToString());
				Debug.Assert(i2 == i4);
			}
#endif
		}

		public void ComputeAlternativeChoiceFunction(Vector<int>[] x_F1, Vector<int>[]? mask)
		{
			Debug.Assert((x_F1.LongLength % 2) == 0);
			Debug.Assert((mask == null) || (x_F1.LongLength == mask.LongLength * 2));
			Debug.Assert(x_F1.LongLength == Common.SimdLengthEncoded<int>(_inputLen));

			long diffSum = 0;
			var inputLenSimd = x_F1.LongLength;
			var dSimd = inputLenSimd >> 1;

			if(mask == null) {
				var d = _inputLen >> 1;
				var diffSumLow = Vector<long>.Zero;
				var diffSumHigh = Vector<long>.Zero;

				for(long i = 0; i < dSimd; ++i) {
					// equal to Abs(Min(x, w) - w); i.e., the |x∧w - w| form of the definition
					Vector<int> diffVec = Vector.Max(_weights[i] - x_F1[i], Vector<int>.Zero) +
												Vector.Max(_weights[i + dSimd] - x_F1[i + dSimd], Vector<int>.Zero);
					Vector.Widen(diffVec, out Vector<long> low, out Vector<long> high);
					diffSumLow += low;
					diffSumHigh += high;
				}
				diffSum = Common.HorizontalSum(diffSumLow + diffSumHigh);

				Activation = (int)(Common.ScalingFactor - diffSum / d);
			} else {
				long diffNum = 0;
				Vector<int> maskDiffVec = Vector<int>.Zero;
				var diffSumLow = Vector<long>.Zero;
				var diffSumHigh = Vector<long>.Zero;

				for(long i = 0; i < dSimd; ++i) {
					// masked variant of Max(w - x, 0); see the comment in the unmasked branch above
					var diffVec = Vector.BitwiseAnd(Vector.Max(_weights[i] - x_F1[i], Vector<int>.Zero), mask[i]) +
												Vector.BitwiseAnd(Vector.Max(_weights[i + dSimd] - x_F1[i + dSimd], Vector<int>.Zero), mask[i]);
					maskDiffVec += mask[i];
					Vector.Widen(diffVec, out Vector<long> low, out Vector<long> high);
					diffSumLow += low;
					diffSumHigh += high;
				}
				diffSum = Common.HorizontalSum(diffSumLow + diffSumHigh);

				for(var j = 0; j < Vector<int>.Count; ++j)
					diffNum -= maskDiffVec[j];

				Activation = (int)((diffNum == 0) ? -Common.ScalingFactor :
								   (Common.ScalingFactor - diffSum / diffNum));
			}
			_matchValue = (int)-Common.ScalingFactor;
		}

		public virtual void ComputeChoiceAndMatchFunction(Vector<int>[] x_F1, long[]? t_F1, Vector<int>[]? mask, int alpha)
		{
			Debug.Assert(t_F1 == null);
			ComputeSpatialChoiceAndMatchFunction(x_F1, mask, alpha);
		}

		protected void ComputeSpatialChoiceAndMatchFunction(Vector<int>[] x_F1, Vector<int>[]? mask, int alpha)
		{
			Debug.Assert((x_F1.LongLength % 2) == 0);
			Debug.Assert((mask == null) || (x_F1.LongLength == mask.LongLength * 2));
			Debug.Assert(x_F1.LongLength == _weights.LongLength);
			Debug.Assert(x_F1.LongLength == Common.SimdLengthEncoded<int>(_inputLen));

			var inputLenSimd = x_F1.LongLength;
			var dSimd = inputLenSimd >> 1;

			var minSumLow = Vector<long>.Zero;
			var minSumHigh = Vector<long>.Zero;

			if(_weightsSum == UNDEFINED) {
				var weightsSumLow = Vector<long>.Zero;
				var weightsSumHigh = Vector<long>.Zero;

				for(long i = 0; i < dSimd; ++i) {
					Vector<int> weightsVec = _weights[i] + _weights[i + dSimd];
					Vector<int> minVec = Vector.Min(x_F1[i], _weights[i]) + Vector.Min(x_F1[i + dSimd], _weights[i + dSimd]);
					Vector.Widen(weightsVec, out Vector<long> low, out Vector<long> high);
					weightsSumLow += low;
					weightsSumHigh += high;
					Vector.Widen(minVec, out low, out high);
					minSumLow += low;
					minSumHigh += high;
				}

				_weightsSum = Common.HorizontalSum(weightsSumLow + weightsSumHigh);
			} else {
				for(long i = 0; i < dSimd; ++i) {
					var minVec = Vector.Min(x_F1[i], _weights[i]) + Vector.Min(x_F1[i + dSimd], _weights[i + dSimd]);
					Vector.Widen(minVec, out Vector<long> low, out Vector<long> high);
					minSumLow += low;
					minSumHigh += high;
				}
			}

			long minSum = Common.HorizontalSum(minSumLow + minSumHigh);

			var normedMinSum = (minSum / _inputLen) * Common.ScalingFactor;		// scale to constrain the maximum value
			Activation = (int)(normedMinSum / (alpha + _weightsSum) * _inputLen);  	// rescale

			if(mask == null) {
				long x_F1_sum	=	(_inputLen >> 1) * Common.ScalingFactor;
				_matchValue 	=	(int)(normedMinSum / x_F1_sum * _inputLen);
			} else {
				long partial_x_F1_sum_false = 0;
				long partial_x_F1_sum_true = 0;
				Vector<int> falseVec = Vector<int>.Zero;
				Vector<int> trueVec = Vector<int>.Zero;
				var falseMinSumLow = Vector<long>.Zero;
				var falseMinSumHigh = Vector<long>.Zero;
				var trueMinSumLow = Vector<long>.Zero;
				var trueMinSumHigh = Vector<long>.Zero;

				for(long i = 0; i < dSimd; ++i) {
					var maskComplement = Vector.OnesComplement(mask[i]);

					falseVec += mask[i];
					trueVec += maskComplement;

					var minVec1 = Vector.Min(x_F1[i], _weights[i]);
					var minVec2 = Vector.Min(x_F1[i + dSimd], _weights[i + dSimd]);
					var falseMinVec = Vector.BitwiseAnd(minVec1, mask[i]) +
													Vector.BitwiseAnd(minVec2, mask[i]);
					var trueMinVec = Vector.BitwiseAnd(minVec1, maskComplement) +
													Vector.BitwiseAnd(minVec2, maskComplement);

					Vector.Widen(falseMinVec, out Vector<long> low, out Vector<long> high);
					falseMinSumLow += low;
					falseMinSumHigh += high;
					Vector.Widen(trueMinVec, out low, out high);
					trueMinSumLow += low;
					trueMinSumHigh += high;
				}

				long partialMinSumFalse = Common.HorizontalSum(falseMinSumLow + falseMinSumHigh);
				long partialMinSumTrue = Common.HorizontalSum(trueMinSumLow + trueMinSumHigh);

				for(var j = 0; j < Vector<int>.Count; ++j) {
					partial_x_F1_sum_false -= falseVec[j];
					partial_x_F1_sum_true -= trueVec[j];
				}

				// correct for extra ints
				partial_x_F1_sum_true -= Vector<int>.Count * dSimd - (_inputLen >> 1);

				partial_x_F1_sum_false *= Common.ScalingFactor;
				partial_x_F1_sum_true *= Common.ScalingFactor;

				long partialMatchValueFalse;
				long partialMatchValueTrue;

				if(partial_x_F1_sum_false > 0) {
					var normedPartialMinSumFalse = (partialMinSumFalse / _inputLen) * Common.ScalingFactor;		// scale to constrain the maximum value
					partialMatchValueFalse = normedPartialMinSumFalse / partial_x_F1_sum_false * _inputLen;	// rescale
				} else
					partialMatchValueFalse = Common.ScalingFactor;

				if(partial_x_F1_sum_true > 0) {
					var normedPartialMinSumTrue = (partialMinSumTrue / _inputLen) * Common.ScalingFactor;		// scale to constrain the maximum value
					partialMatchValueTrue = normedPartialMinSumTrue / partial_x_F1_sum_true * _inputLen;	// rescale
				} else
					partialMatchValueTrue = Common.ScalingFactor;

				_matchValue = (int)Math.Min(partialMatchValueFalse, partialMatchValueTrue);
			}
		}

		public virtual int GetCombinedChoiceAndMatchValue(MatchFunction<FTA_F2_node, long>? match, long rho)
		{
			return ((match != null) && match(this, rho) ? Activation : (int)-Common.ScalingFactor);
		}

		public decimal[] GetCopyOfSpatialWeights()
		{
			var weightsCopy = new decimal[_inputLen];
			var d = _inputLen >> 1;
			var dSimd = _weights.LongLength >> 1;

			long i, i1;
			int i2;
			for(i = 0, i1 = 0, i2 = 0; i < d; ++i) {
				weightsCopy[i] = _weights[i1][i2] / (decimal)Common.ScalingFactor;
				weightsCopy[i + d] = _weights[i1 + dSimd][i2] / (decimal)Common.ScalingFactor;

				if(++i2 == Vector<int>.Count) {
					i2 = 0;
					++i1;
				}
			}

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

		public virtual void SaveText(TextWriter writer)
		{
			writer.WriteLine("------------ F2 node ------------");

			writer.WriteLine("node ID: " + NodeID);
			writer.WriteLine("cluster ID: " + ClusterID);
			writer.WriteLine("input length: " + _inputLen);
			writer.WriteLine("represented inputs: " + _representedInputs);
			writer.WriteLine("activation: " + (Activation / (decimal)Common.ScalingFactor).ToString(CultureInfo.InvariantCulture));
			writer.WriteLine("match value: " + (_matchValue / (decimal)Common.ScalingFactor).ToString(CultureInfo.InvariantCulture));

			SaveAdditionalText(writer);

			writer.WriteLine("............. edges .............");

			SaveEdgesText(writer, Common.TopoART_file_format_version);

			writer.WriteLine("............ weights ............");
			for(long i = 0; i < _inputLen; ++i) {
				if(i != 0)
					writer.Write(" ");
				(long i1, int i2) = Common.SimdIndexes<int>(i, _inputLen >> 1);
				writer.Write((_weights[i1][i2] / (decimal)Common.ScalingFactor).ToString(CultureInfo.InvariantCulture));
			}
			writer.Write("\n");
		}

		protected virtual void SaveAdditionalText(TextWriter writer) {}

		public virtual void Save(BinaryWriter writer, bool compatibilityMode)
		{
			writer.Write(NodeID);
			writer.Write(_inputLen);
			writer.Write(_representedInputs);
			SaveEdges(writer, Common.TopoART_file_format_version);

			if(compatibilityMode) {
				for(long i = 0; i < _inputLen; ++i) {
					var (i1, i2) = Common.SimdIndexes<int>(i, _inputLen >> 1);
					writer.Write(_weights[i1][i2] / (decimal)Common.ScalingFactor);
				}
				writer.Write(Activation / (decimal)Common.ScalingFactor);
				writer.Write(_matchValue / (decimal)Common.ScalingFactor);
			} else {
				for(long i = 0; i < _inputLen; ++i) {
					var (i1, i2) = Common.SimdIndexes<int>(i, _inputLen >> 1);
					writer.Write(_weights[i1][i2]);
				}
				writer.Write(Activation);
				writer.Write(_matchValue);
			}

			writer.Write(ClusterID);

			SaveAdditionalData(writer);
		}

		protected virtual void SaveAdditionalData(BinaryWriter writer) {}
	}

//**********************************************************************************************************************

	internal sealed class FETA_F2_node : FTA_F2_node
	{
		// used only for code simplification
		private const long UNDEFINED = LibTopoART_info.UNDEFINED;

		private long _t_max;
		private readonly long[] _temporalWeights;		// the type of the temporal weights must equal the type of t_max
		private int _temporalMatchValue;
		private int _combinedMatchValue;

//----------------------------------------------------------------------------------------------------------------------

		public decimal[] TemporalWeights
		{
			get => [ ((decimal)_temporalWeights[0] / Common.ScalingFactor), ((decimal)_temporalWeights[1] / Common.ScalingFactor) ];
		}

		public decimal ComputeForwardTemporalDistance(FETA_F2_node otherNode)
		{
			decimal result;

			// ensure forward search
			if(otherNode._temporalWeights[0] > _temporalWeights[0]) {
				result = (otherNode._temporalWeights[0] - _temporalWeights[0] +
						 Math.Abs(otherNode._temporalWeights[1] - _temporalWeights[1])) /
						 (decimal)Common.ScalingFactor;
			} else
				result = UNDEFINED;	// UNDEFINED is negative

			return result;
		}

		public override int MatchValue
		{
			get => _combinedMatchValue;
			set => _combinedMatchValue = value;
		}

//----------------------------------------------------------------------------------------------------------------------

		public FETA_F2_node(long nodeID, long inputLen, long t_max, Vector<int>[] initialSpatialWeights,
			long[] initialTemporalWeights)
			: base(nodeID, inputLen, initialSpatialWeights)
		{
			_t_max = t_max;

			_temporalWeights = [ initialTemporalWeights[0], initialTemporalWeights[1] ];

			_temporalMatchValue	=	(int)-Common.ScalingFactor;
			_combinedMatchValue	=	(int)-Common.ScalingFactor;
		}

		public FETA_F2_node(BinaryReader reader, (FileFormatVersions fileFormatVersions, bool compatibilityMode) fileFormatInfo, long t_max)
			: base(reader, fileFormatInfo)
		{
			if(fileFormatInfo.fileFormatVersions.FileFormatVersion == 0.01m)
				_t_max = reader.ReadInt64() * Common.ScalingFactor;	// saved as long
			else if (fileFormatInfo.fileFormatVersions.FileFormatVersion >= 0.02m)
				_t_max = t_max;
			else
				throw new InvalidFileException(Common.InvalidFileException_InvalidVersion);

			_temporalWeights = new long[2];

			if(fileFormatInfo.compatibilityMode) {
				_temporalMatchValue	=	(int)(reader.ReadDecimal() * Common.ScalingFactor);
				_combinedMatchValue	=	(int)(reader.ReadDecimal() * Common.ScalingFactor);
				_temporalWeights[0]	=	(long)(reader.ReadDecimal() * Common.ScalingFactor);
				_temporalWeights[1]	=	(long)(reader.ReadDecimal() * Common.ScalingFactor);
			} else {
				_temporalMatchValue	=	reader.ReadInt32();
				_combinedMatchValue	=	reader.ReadInt32();
				_temporalWeights[0]	=	reader.ReadInt64();
				_temporalWeights[1]	=	reader.ReadInt64();
			}
		}

//----------------------------------------------------------------------------------------------------------------------

		public override void ComputeChoiceAndMatchFunction(Vector<int>[] x_F1, long[]? t_F1, Vector<int>[]? mask, int alpha)
		{
			Debug.Assert(t_F1 != null);

			_temporalMatchValue = (t_F1 != null) ? (int)((_t_max - Math.Min(t_F1[1] - _temporalWeights[0], _t_max)) /
										   (_t_max / Common.ScalingFactor)) : (int)LibTopoART_info.UNDEFINED;

			if(_temporalMatchValue != 0)
				ComputeSpatialChoiceAndMatchFunction(x_F1, mask, alpha);
			else
			{
				// Do not compute spatial activation and vigilance criterion to save processing time.
				Activation = 0;
				_matchValue = 0;
			}

			// combine spatial and temporal similarity
			_combinedMatchValue = Math.Min(_matchValue, _temporalMatchValue);
		}

		public void AdaptWeights(Vector<int>[] x_F1, long[] t_F1, int beta)
		{
			AdaptWeights(x_F1, beta);

			if(beta == (int)Common.ScalingFactor)
				_temporalWeights[1] = Math.Max(t_F1[1], _temporalWeights[1]);
			else if(beta != 0)
				_temporalWeights[1] = (long)((beta * (decimal)(Math.Max(t_F1[1], _temporalWeights[1]))
					+ (Common.ScalingFactor - beta) * (decimal)_temporalWeights[1]) / Common.ScalingFactor);
		}

		public override decimal[] GetCopyOfTemporalWeights()
		{
			return [ TemporalWeights[0], TemporalWeights[1] ];
		}

//----------------------------------------------------------------------------------------------------------------------

		public void PrintSpatialWeights()
		{
			PrintWeights();
		}

		public void PrintTemporalWeights()
		{
			Console.WriteLine("{0} {1}", TemporalWeights[0], TemporalWeights[1]);
		}

//----------------------------------------------------------------------------------------------------------------------

		public override void SaveText(TextWriter writer)
		{
			base.SaveText(writer);

			writer.WriteLine("......... temporal data .........");
			writer.WriteLine("temporal match value: " + (_temporalMatchValue / (decimal)Common.ScalingFactor).ToString(CultureInfo.InvariantCulture));
			writer.WriteLine("combined match value: " + (_combinedMatchValue / (decimal)Common.ScalingFactor).ToString(CultureInfo.InvariantCulture));

			writer.WriteLine("....... temporal weights ........");
			writer.Write(TemporalWeights[0].ToString(CultureInfo.InvariantCulture));
			writer.Write(" ");
			writer.Write(TemporalWeights[1].ToString(CultureInfo.InvariantCulture));

			writer.Write("\n");
		}

		public override void Save(BinaryWriter writer, bool compatibilityMode)
		{
			base.Save(writer, compatibilityMode);

#pragma warning disable 162
			if(Common.Episodic_TopoART_file_format_version == 0.01m)
				writer.Write(_t_max / Common.ScalingFactor);	// no decimal necessary, integer number
#pragma warning restore 162

			if(compatibilityMode) {
				writer.Write(_temporalMatchValue / (decimal)Common.ScalingFactor);
				writer.Write(_combinedMatchValue / (decimal)Common.ScalingFactor);
				writer.Write(_temporalWeights[0] / (decimal)Common.ScalingFactor);
				writer.Write(_temporalWeights[1] / (decimal)Common.ScalingFactor);
			} else {
				writer.Write(_temporalMatchValue);
				writer.Write(_combinedMatchValue);
				writer.Write(_temporalWeights[0]);
				writer.Write(_temporalWeights[1]);
			}
		}
	}

//**********************************************************************************************************************

	internal sealed class FTAC_F2_node : FTA_F2_node
	{
		private long _classID;
		private long _sizeCache = LibTopoART_info.UNDEFINED;

		public override long ClassID { get => _classID; }

		public long Size
		{
			get {

				if(_sizeCache == LibTopoART_info.UNDEFINED) {
					var dSimd = _weights.LongLength >> 1;

					Debug.Assert(dSimd == Common.SimdLength<int>(_inputLen >> 1));

					var sizeLow = Vector<long>.Zero;
					var sizeHigh = Vector<long>.Zero;
					for(long i = 0; i < dSimd; ++i) {
						var absVec = Vector.Abs((Common.ScalingVectorInt - _weights[dSimd + i]) - _weights[i]);
						Vector.Widen(absVec, out Vector<long> low, out Vector<long> high);
						sizeLow += low;
						sizeHigh += high;
					}
					long size = Common.HorizontalSum(sizeLow + sizeHigh);

					// correction for the additional elements of the final Vector<int>
					size -= (dSimd * Vector<int>.Count - (_inputLen >> 1)) * Common.ScalingFactor;

					_sizeCache = size;
				}

				return _sizeCache;
			}
		}

//----------------------------------------------------------------------------------------------------------------------

		public FTAC_F2_node(long nodeID, long input_len, Vector<int>[] initialWeights, long classID)
			: base(nodeID, input_len, initialWeights)
		{
			_classID = classID;
		}

		public FTAC_F2_node(BinaryReader reader, in (FileFormatVersions, bool) fileFormatInfo)
			: base(reader, fileFormatInfo) {}

		protected override void LoadAdditionalData(BinaryReader reader, in (FileFormatVersions, bool) fileFormatInfo)
		{
			_classID = reader.ReadInt64();
		}

//----------------------------------------------------------------------------------------------------------------------

		public override void AdaptWeights(Vector<int>[] x_F1, int beta)
		{
			base.AdaptWeights(x_F1, beta);

			_sizeCache = LibTopoART_info.UNDEFINED;
		}

//----------------------------------------------------------------------------------------------------------------------

		protected override void SaveAdditionalText(TextWriter writer)
		{
			writer.WriteLine("class ID: " + ClassID);
		}

		protected override void SaveAdditionalData(BinaryWriter writer)
		{
			writer.Write(ClassID);
		}
	}

//**********************************************************************************************************************

}