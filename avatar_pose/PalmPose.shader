Shader "DesktopBody/PalmPose" {
 Properties { _Side ("Right hand", Float) = 0 }
 SubShader {
  Tags { "Queue"="Overlay+101" "DisableBatching"="True" "IgnoreProjector"="True" }
  Pass {
   Cull Off ZWrite Off ZTest Always
   CGPROGRAM
   #pragma vertex vert
   #pragma fragment frag
   #pragma target 4.0
   #pragma multi_compile_instancing
   #include "UnityCG.cginc"
   float _Side, _VRChatCameraMode, _VRChatMirrorMode;
   struct Input { float4 vertex:POSITION; float2 uv:TEXCOORD0; UNITY_VERTEX_INPUT_INSTANCE_ID };
   struct Output { float4 pos:SV_POSITION; float2 uv:TEXCOORD0; UNITY_VERTEX_INPUT_INSTANCE_ID UNITY_VERTEX_OUTPUT_STEREO };
   Output vert(Input v) {
    UNITY_SETUP_INSTANCE_ID(v);
    Output o; UNITY_INITIALIZE_OUTPUT(Output,o); UNITY_TRANSFER_INSTANCE_ID(v,o); UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
    o.uv=v.uv;
    float2 screen=float2(.25+.21*_Side,.06)+v.uv*float2(.20,.08);
    o.pos=float4(screen.x*2-1,1-screen.y*2,.5,1); return o;
   }
   uint word(uint i) {
    if(i==0) return _Side>.5 ? 0x50414C52u : 0x50414C4Cu;
    if(i<4) return asuint(unity_ObjectToWorld[i-1][3]);
    uint axis=(i-4)/3;
    float3 dir=normalize(float3(unity_ObjectToWorld[0][axis],unity_ObjectToWorld[1][axis],unity_ObjectToWorld[2][axis]));
    return asuint(dir[(i-4)%3]);
   }
   float4 frag(Output p):SV_Target {
    UNITY_SETUP_INSTANCE_ID(p); UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(p);
    if(_VRChatCameraMode!=0 || _VRChatMirrorMode!=0) discard;
    uint row=min((uint)(p.uv.y*14),13u), bits;
    if(row==13) { bits=2166136261u; [unroll] for(uint i=0;i<13;i++) bits=(bits^word(i))*16777619u; }
    else bits=word(row);
    float bit=(bits>>min((uint)(p.uv.x*32),31u))&1u;
    return float4(bit,bit,bit,1);
   }
   ENDCG
  }
 }
}
