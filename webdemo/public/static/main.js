$(document).ready(function () {
  let file = null;

  function submit(e) {
    $("#iuv-image").hide()
    $("#loader-spinner").show()
    let data = new FormData($("#main-form")[0]);
    $.ajax({
      type: "POST",
      url: "/upload",
      data: data,
      enctype: "multipart/form-data",
      processData: false,
      contentType: false,
      cache: false,
      success: function (data) {
        $("#loader-spinner").hide();
        let image = $("#iuv-image");
        image[0].src = "data:image/png;base64," + data.iuv;
        image.show();
        console.log(data);
      },
      error: function (data) {
        $("#loader-spinner").hide();
        console.log(data);
      },
    });
  }

  $('input[name="file1"]').change(function (e) {
    if (e.target.files) {
      let reader = new FileReader(),
          image = $("#input-image");
      reader.onloadend = function () {
        image[0].src = reader.result;
        image.show();
      };
      file = e.target.files[0];
      reader.readAsDataURL(file);
    }
    if (file)
      submit();
  });
});
