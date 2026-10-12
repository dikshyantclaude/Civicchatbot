(() => {
 const token = document.querySelector('meta[name="csrf-token"]').content;
 const originalFetch = window.fetch;
 window.fetch = function(input, options = {}) {
   const url = new URL(typeof input === 'string' ? input : input.url, location.href);
   if (url.origin === location.origin) {
     options = {...options, headers: new Headers(options.headers || (input instanceof Request ? input.headers : {}))};
     options.headers.set('X-CSRF-Token', token);
   }
   return originalFetch.call(this, input, options);
 };
 const open = XMLHttpRequest.prototype.open, send = XMLHttpRequest.prototype.send;
 XMLHttpRequest.prototype.open = function(method,url,...rest) {this.civicSameOrigin = new URL(url,location.href).origin === location.origin; return open.call(this,method,url,...rest);};
 XMLHttpRequest.prototype.send = function(body) {if(this.civicSameOrigin)this.setRequestHeader('X-CSRF-Token',token);return send.call(this,body);};
 window.escapeHTML = value => String(value ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
})();
