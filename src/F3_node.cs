namespace LibTopoART
{

//**********************************************************************************************************************

	internal class F3_node
	{
		// used only for code simplification
		private const long UNDEFINED = LibTopoART_info.UNDEFINED;
		public F3_node? _next;

//----------------------------------------------------------------------------------------------------------------------

		public long Activation { get; }
		public FTA_F2_node? BestMatchingF2Node { get; }
		public long ClusterID { get; }

//----------------------------------------------------------------------------------------------------------------------

		public F3_node(FTA_F2_node? bm_node)
		{
			if(bm_node != null) {
				Activation		=	bm_node.Activation;
				ClusterID		=	bm_node.ClusterID;
			} else {
				Activation		=	UNDEFINED;
				ClusterID		=	UNDEFINED;
			}
			BestMatchingF2Node	=	bm_node;
		}
	}

//**********************************************************************************************************************

}