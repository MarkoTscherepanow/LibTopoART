using System.IO;
using System.Collections.Generic;

namespace LibTopoART
{
#pragma warning disable 1591

//**********************************************************************************************************************

#if DEBUG
	/// <summary>
	/// Class containing the edges of F2 nodes.
	/// </summary>
	public class F2_edges
#else
	internal class F2_edges
#endif
	{
		private const long _minCapacity = 8;
		private long _capacity;
		private long[]? _connectedNodeIDs;

		public long EdgeNum { get; private set; }
		public long StartingNodeID { get; private set; }

		public List<(long, long)> Edges
		{
			get {
				if(EdgeNum > int.MaxValue)
					throw new InvalidNumberException(Common.InvalidNumberException_EdgeNumberTooHigh);

				var list = new List<(long, long)>((int)EdgeNum);
				if(_connectedNodeIDs != null) {
					for(long i = 0; i < EdgeNum; ++i)
						list.Add((StartingNodeID, _connectedNodeIDs[i]));
				}
				return list;
			}
		}

		public void InitEdges(long startingNodeID, long capacity = _minCapacity)
		{
			_capacity = capacity;
			EdgeNum = 0;
			StartingNodeID = startingNodeID;
			_connectedNodeIDs = new long[capacity];
		}

		public void InitEdges(long startingNodeID, long edgeNum, BinaryReader reader, in decimal taFileFormatVersion)
		{
			InitEdges(startingNodeID, edgeNum + _minCapacity);

			if(taFileFormatVersion == 0.09m) {
				for(long i = 0; i < edgeNum; ++i) {
					var a = reader.ReadInt64();
					var b = reader.ReadInt64();
					if(a == StartingNodeID)
						AddEdgeTo(b);
					else
						throw new InvalidFileException(Common.InvalidFileException_FileCorrupted);
				}
			}
			else if(taFileFormatVersion >= 0.10m) {
				for(long i = 0; i < edgeNum; ++i) {
					var b = reader.ReadInt64();
					AddEdgeTo(b);
				}
			}
			else
				throw new InvalidFileException(Common.InvalidFileException_InvalidVersion);
		}

		public bool AddEdgeTo(long connectedNodeID)
		{
			if(!ExistsEdgeTo(connectedNodeID)) {
				if(EdgeNum >= _capacity)
					Grow();
				_connectedNodeIDs![EdgeNum] = connectedNodeID;
				++EdgeNum;
				return true;
			}

			return false;
		}

		public bool ExistsEdgeTo(long connectedNodeID)
		{
			if(_connectedNodeIDs != null) {
				for(long i = EdgeNum - 1; i >= 0; --i)
					if(_connectedNodeIDs[i] == connectedNodeID)
						return true;
			}
			return false;
		}

		public void GetConnectedNodeIDs(out long size, out long[]? connectedNodeIDs)
		{
			size = EdgeNum;
			connectedNodeIDs = _connectedNodeIDs;
		}

		private void Grow()
		{
			_capacity <<= 1;
			var newArray = new long[_capacity];
			if(_connectedNodeIDs != null) {
				for(long i = 0; i < EdgeNum; ++i)
					newArray[i] = _connectedNodeIDs[i];
			}
			_connectedNodeIDs = newArray;
		}

		public bool RemoveEdgeTo(long connectedNodeID)
		{
			if(_connectedNodeIDs != null) {
				for(long i = EdgeNum - 1; i >= 0; --i)
					if(_connectedNodeIDs[i] == connectedNodeID) {
						for(long j = i + 1; j < EdgeNum; ++j)
							_connectedNodeIDs[j - 1] = _connectedNodeIDs[j];
						--EdgeNum;
						return true;
					}
			}
			return false;
		}

		public void SaveEdges(BinaryWriter writer, in decimal TopoART_file_format_version)
		{
			writer.Write(_connectedNodeIDs != null ? EdgeNum : 0);

			if(_connectedNodeIDs != null) {
#if DEBUG
				if(TopoART_file_format_version == 0.09m) {
					for(var i = EdgeNum - 1; i >= 0; --i) {
						writer.Write(StartingNodeID);
						writer.Write(_connectedNodeIDs[i]);
					}
				} else
#endif
					if(TopoART_file_format_version >= 0.10m) {
						for(var i = EdgeNum - 1; i >= 0; --i)
							writer.Write(_connectedNodeIDs[i]);
					}
			}
		}

		public void SaveEdgesText(TextWriter writer, in decimal TopoART_file_format_version)
		{
#if DEBUG
			if(TopoART_file_format_version == 0.09m) {
				if(_connectedNodeIDs != null)
					for(var i = EdgeNum - 1; i >= 0; --i)
						writer.WriteLine(StartingNodeID + " -> " + _connectedNodeIDs[i]);
			} else
#endif
				if(TopoART_file_format_version >= 0.10m) {
					if(_connectedNodeIDs != null) {
						writer.WriteLine("edge number: " + EdgeNum);
						writer.Write("edge targets: ");
						for(var i = EdgeNum - 1; i >= 0; --i) {
							writer.Write(_connectedNodeIDs[i]);
							if(i > 0)
								writer.Write(" ");
						}
					} else {
						writer.WriteLine("edge number: " + 0);
						writer.Write("edge targets: ");
					}

					writer.Write("\n");
				}
		}
	}

//**********************************************************************************************************************

#pragma warning restore 1591
}