import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import xarray


def analysis(file: str = "../../tests/data/labels/t2m_cerra_201801_005deg.nc"):
    ds = xarray.open_dataset(file)

    projection = ccrs.Robinson()
    crs = ccrs.PlateCarree()

    plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=projection, frameon=True)
    gl = ax.gridlines(
        crs=crs,
        draw_labels=True,
        linewidth=0.6,
        color="gray",
        alpha=0.5,
        linestyle="-.",
    )
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    ax.add_feature(cf.COASTLINE.with_scale("50m"), lw=0.5)

    cbar_kwargs = {
        "orientation": "horizontal",
        "shrink": 0.6,
        "pad": 0.05,
        "aspect": 40,
        "label": "Temperature",
    }

    lon_min = -50
    lon_max = 60
    lat_min = 10
    lat_max = 80
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=crs)

    ds.mean("time")["t2m"].plot(
        ax=ax,
        cmap="YlOrBr",
        transform=ccrs.PlateCarree(),
        cbar_kwargs=cbar_kwargs,
        vmin=250,
        vmax=300,
    )

    plt.title("Mean temperature January 2018 - CERRA")
    plt.close()


if __name__ == "__main__":
    analysis()
