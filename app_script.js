// <script defer src="https://busuanzi.9420.ltd/js"></script>

// <script async src="//busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js"></script>
// var dynamicURL ="https://busuanzi.9420.ltd/js";
// var src=GetText(dynamicURL);
// eval(src);
const script = document.createElement("script");
script.onload = () =>  console.log("busuanzi loaded") ;
script.src = "https://busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js";
document.head.appendChild(script)

// replaceFooter
document.getElementsByTagName("footer")[0].innerHTML =
  // 'Powered by &nbsp <a href="https://github.com/mit-han-lab/llm-awq/tree/main/tinychat" style="color:black">TinyChat</a> &nbsp with 4-bit &nbsp <a href="https://github.com/mit-han-lab/llm-awq" style="color:black">AWQ</a>.';
 'Total Clicks: <span id="busuanzi_value_site_pv"></span>';

// force light mode
// https://github.com/gradio-app/gradio/issues/7384#issuecomment-1937898519
// const url = new URL(window.location);
// if (url.searchParams.get("__theme") !== "light") {
//   url.searchParams.set("__theme", "light");
//   window.location.href = url.href;
// }
